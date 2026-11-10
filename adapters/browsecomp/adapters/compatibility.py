from contextlib import ExitStack, contextmanager

_MISSING = object()


@contextmanager
def patched_attributes(*bindings):

    def restore(target, name, original):
        if original is _MISSING:
            delattr(target, name)
        else:
            setattr(target, name, original)

    with ExitStack() as stack:
        for target, name, replacement in bindings:
            original = getattr(target, name, _MISSING)
            setattr(target, name, replacement)
            stack.callback(restore, target, name, original)
        yield
