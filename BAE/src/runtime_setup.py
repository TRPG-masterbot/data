"""출력 보조 파일을 준비합니다. 모델·GPU·팀 코드는 변경하지 않습니다."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib
import json
import linecache
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import time
from urllib.parse import quote, urlsplit
from urllib.request import urlopen


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("배포 목록에 중복된 이름이 있습니다: " + str(key))
        result[key] = value
    return result


def _relative_name(name):
    # Windows 경로·상위 폴더·URL 조각을 허용하지 않습니다.
    if (not isinstance(name, str) or not name or any(c in name for c in "\\:%?#\x00")
            or any(ord(c) < 32 for c in name) or name.startswith("/")
            or any(part in {"", ".", ".."} for part in name.split("/"))
            or PurePosixPath(name).as_posix() != name):
        raise ValueError("올바르지 않은 보조 파일 경로입니다: " + repr(name))
    return name


def _destination(root, name):
    target = (root / _relative_name(name)).resolve()
    if not target.is_relative_to(root):
        raise ValueError("출력 폴더 밖의 경로는 사용할 수 없습니다.")
    return target


def _matches(path, entry):
    return path.is_file() and path.stat().st_size == entry["bytes"] and _digest(path.read_bytes()) == entry["sha256"]


def _atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".bae-", suffix=".tmp", dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_url(url):
    with urlopen(url, timeout=60) as response:
        return response.read()


def _invalidate_previous(previous_root, root):
    if previous_root is None or previous_root == root:
        return []
    own_names = {"gameplay", "game_ui", "play_view", "portrait_view", "ollama_backend",
                 "action_requirements", "state_projection", "diagnostic_export"}
    removed = []
    for name, module in list(sys.modules.items()):
        if name not in own_names and name != "game" and not name.startswith("game."):
            continue
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().is_relative_to(previous_root):
            del sys.modules[name]
            removed.append(name)
    return removed


def prepare_runtime(base_url, manifest_sha, target=None, previous_root=None):
    """확인된 목록의 파일만 준비하고 (폴더, 준비 기록)을 반환합니다.

    같은 목록과 파일이 있으면 통신하거나 다시 쓰지 않습니다. 파일 하나가
    손상된 경우에는 그 파일만 다시 받고, 새 버전은 별도 폴더를 사용합니다.
    """
    started = time.perf_counter()
    if not isinstance(manifest_sha, str) or re.fullmatch(r"[0-9a-f]{64}", manifest_sha) is None:
        raise ValueError("배포 목록의 SHA256 값이 필요합니다.")
    url = urlsplit(str(base_url))
    if (url.scheme not in {"http", "https"} or not url.netloc or url.username or url.password
            or url.query or url.fragment):
        raise ValueError("보조 파일의 http 또는 https 주소를 확인하세요.")
    base_url = str(base_url).rstrip("/") + "/"
    default = (Path("/content") if Path("/content").is_dir() else Path.cwd())
    root = (Path(target) if target is not None else default / "BAE" / "versions" / manifest_sha[:16] / "src").resolve()
    previous = Path(previous_root).resolve() if previous_root is not None else None
    package_root = Path(__file__).resolve().parent
    cache_path = root / "runtime-manifest.json"
    cached_raw = cache_path.read_bytes() if cache_path.is_file() else None
    cached_manifest = cached_raw is not None and _digest(cached_raw) == manifest_sha
    package_manifest = package_root / "runtime-manifest.json"
    package_raw = package_manifest.read_bytes() if package_manifest.is_file() else None
    local_manifest = package_raw is not None and _digest(package_raw) == manifest_sha
    raw = (cached_raw if cached_manifest else package_raw if local_manifest
           else _read_url(base_url + "runtime-manifest.json"))
    if _digest(raw) != manifest_sha:
        raise ValueError("배포 목록의 SHA256 값이 일치하지 않습니다.")
    manifest = json.loads(raw.decode("utf-8"), object_pairs_hook=_json_object)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise ValueError("배포 파일 목록이 비어 있거나 형식이 다릅니다.")
    entries = manifest["files"]
    # 전체 목록과 최종 경로를 먼저 검사합니다. 오류가 있으면 파일을 쓰지 않습니다.
    for name, entry in entries.items():
        _destination(root, name)
        if name == "runtime-manifest.json":
            raise ValueError("배포 목록 자체는 파일 목록에 넣을 수 없습니다.")
        if (not isinstance(entry, dict) or re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))) is None
                or type(entry.get("bytes")) is not int or entry["bytes"] < 0):
            raise ValueError("보조 파일의 SHA256 또는 크기가 올바르지 않습니다: " + name)

    root.mkdir(parents=True, exist_ok=True)
    info = {"manifest_sha256": manifest_sha, "version": manifest.get("version"),
            "manifest_cache_hit": cached_manifest, "cache_hits": 0, "files_fetched": 0,
            "manifest_source": "cache" if cached_manifest else "local_package" if local_manifest else "remote",
            "self_copied": 0, "files_copied_from_package": 0, "files_reused_from_previous": 0}
    pending = []
    for name, entry in entries.items():
        path = _destination(root, name)
        if _matches(path, entry):
            info["cache_hits"] += 1
            continue
        local = _destination(package_root, name)
        if _matches(local, entry):
            _atomic_write(path, local.read_bytes())
            info["files_copied_from_package"] += 1
            if name == "runtime_setup.py":
                info["self_copied"] += 1
            continue
        previous_path = _destination(previous, name) if previous is not None and previous != root else None
        if previous_path is not None and _matches(previous_path, entry):
            _atomic_write(path, previous_path.read_bytes())
            info["files_reused_from_previous"] += 1
            continue
        pending.append((name, entry))

    def download(item):
        name, entry = item
        data = _read_url(base_url + quote(name, safe="/._-"))
        if len(data) != entry["bytes"] or _digest(data) != entry["sha256"]:
            raise ValueError("보조 파일의 SHA256 또는 크기가 다릅니다: " + name)
        _atomic_write(_destination(root, name), data)
        return name

    if pending:
        with ThreadPoolExecutor(max_workers=min(6, len(pending))) as pool:
            for future in as_completed([pool.submit(download, item) for item in pending]):
                future.result()
                info["files_fetched"] += 1
    if not cached_manifest:
        _atomic_write(cache_path, raw)
    info["invalidated_modules"] = _invalidate_previous(previous, root)
    # 이전 보조 폴더만 경로에서 빼고 새 출력 폴더를 우선합니다.
    old_path = str(previous) if previous is not None and previous != root else None
    sys.path[:] = [value for value in sys.path if value != str(root) and value != old_path]
    sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    info["seconds"] = round(time.perf_counter() - started, 3)
    return root, info


def load_support(namespace, name):
    """보조 함수도 원래 노트북의 같은 전역 공간에서 정의합니다."""
    if not isinstance(namespace, dict):
        raise TypeError("노트북 globals()가 필요합니다.")
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
        raise ValueError("보조 코드 이름은 영문·숫자·밑줄로 작성하세요.")
    root = Path(namespace["BAE_RUNTIME_PATH"]).resolve()
    path = _destination(root, "output_helpers/" + name + ".py")
    source = path.read_text(encoding="utf-8")
    linecache.cache[str(path)] = (len(source), None, source.splitlines(True), str(path))
    exec(compile(source, str(path), "exec"), namespace)
