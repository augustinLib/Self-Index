from http import HTTPStatus

REQUEST_ATTEMPTS = 4
RETRY_DELAY_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 7200


class RequestRejected(RuntimeError): ...


def permanent_client_error(status):
    return (
        isinstance(status, int)
        and HTTPStatus.BAD_REQUEST <= status < HTTPStatus.INTERNAL_SERVER_ERROR
        and status not in (HTTPStatus.REQUEST_TIMEOUT, HTTPStatus.TOO_MANY_REQUESTS)
    )


def retry_delay(attempt):
    return RETRY_DELAY_SECONDS * (attempt + 1)
