import dill


class PickleableObject(object):
    """Simple super class to make any python object easily pickable to disc.

    """

    def __init__(self):
        pass

    def save(self, path):
        """Save object to disc.

        Args:
            path (:obj:`str`): File path at which to save, ending with the desired filename.

        """
        with open(path, "wb") as f:
            dill.dump(self, f, dill.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        """Load object from disc.

        Args:
            path (:obj:`str`): File path at which to load, ending with the desired filename.

        .. warning::
            This function will unpickle data from the provied path. The pickle module
            is not intended to be secure against erroneous or maliciously constructed data.
            Never unpickle data received from an untrusted or unauthenticated source.

        """
        with open(path, 'rb') as f:
            return dill.load(f)
