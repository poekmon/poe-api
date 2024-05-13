from typing import Any


class SelfDefinedException(Exception):
    def __init__(self, reason: Any = None, message: str = "", code: int = -1) -> None:
        self.reason = reason  # 异常主要原因
        self.message = message  # 更细节的描述
        self.code = code  # 错误码：-1 为默认；0～1000 以内正数为 http 错误码；10000 以上为自定义错误码

    def __str__(self):
        return f"{self.__class__.__name__}: [{self.code}] {self.reason} {self.message}"


class AuthenticationFailedException(SelfDefinedException):
    def __init__(self, message: str = ""):
        super().__init__(reason="errors.authenticationFailed", message=message, code=401)


class AuthorityDenyException(SelfDefinedException):
    def __init__(self, message: str = ""):
        super().__init__(reason="errors.authorityDeny", message=message, code=403)


class ResourceNotFoundException(SelfDefinedException):
    def __init__(self, message: str = ""):
        super().__init__(reason="errors.resourceNotFound", message=message, code=404)


class InvalidRequestException(SelfDefinedException):
    def __init__(self, message: str = ""):
        super().__init__(reason="errors.invalidRequest", message=message, code=400)


class InternalException(SelfDefinedException):
    def __init__(self, message: str = ""):
        super().__init__(reason="errors.internal", message=message)
