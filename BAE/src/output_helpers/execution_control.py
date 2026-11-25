# 출력 실행 보조 함수: 노트북과 같은 전역 공간에서 원본 본문을 실행합니다.

class OutputStreamCancelled(RuntimeError):
    """종료된 화면·추천 요청의 생성을 중단합니다."""


def _raise_if_cancelled(cancel_event, *, inherited_events=None):
    """이미 끝난 화면 요청이나 취소된 추천 작업은 생성 전에 중단합니다."""
    inherited = _output_stream_cancel_events.get() if inherited_events is None else inherited_events
    for event in (*inherited, cancel_event):
        if event is None:
            continue
        if not callable(getattr(event, "is_set", None)):
            raise TypeError("cancel_event에는 is_set()이 필요합니다.")
        if event.is_set():
            raise OutputStreamCancelled("생성이 취소되었습니다.")


@contextmanager
def _generation_slot(cancel_event=None):
    """생성 차례를 기다리는 동안에도 취소를 확인하고, 끝나면 잠금을 반납합니다."""
    acquired = False
    try:
        while not acquired:
            _raise_if_cancelled(cancel_event)
            acquired = generation_lock.acquire(timeout=.05)
        _raise_if_cancelled(cancel_event)
        yield
    finally:
        if acquired:
            generation_lock.release()


@contextmanager
def output_stream(callback, cancel_event=None):
    """화면 콜백과 취소 신호를 요청에 연결하고, 종료 시 부모 연결을 복원합니다."""
    _raise_if_cancelled(cancel_event)
    inherited = _output_stream_cancel_events.get()
    events = (*inherited, cancel_event) if cancel_event is not None else inherited
    cancel_token = _output_stream_cancel_events.set(events)
    token = _output_stream_sink.set(callback)
    try:
        yield
    finally:
        _output_stream_sink.reset(token)
        _output_stream_cancel_events.reset(cancel_token)


def _emit_output_stream(event):
    """화면에 사본을 보내므로 표시 과정에서 원본 응답이 바뀌지 않습니다."""
    callback = _output_stream_sink.get()
    if callback is not None and callback(deepcopy(event)) is False:
        raise OutputStreamCancelled("화면 요청이 종료되었습니다.")
