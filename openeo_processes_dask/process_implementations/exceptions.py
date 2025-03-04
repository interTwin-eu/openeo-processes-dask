class OpenEOException(Exception):
    pass


class DimensionLabelCountMismatch(OpenEOException):
    pass


class ArrayElementParameterConflict(OpenEOException):
    pass


class ArrayElementParameterMissing(OpenEOException):
    pass


class ArrayNotLabeled(OpenEOException):
    pass


class ArrayElementNotAvailable(OpenEOException):
    pass


class ArrayLabelConflict(OpenEOException):
    pass


class ArrayLengthMismatch(OpenEOException):
    pass


class LabelExists(OpenEOException):
    pass


class TooManyDimensions(OpenEOException):
    pass


class ProcessParameterMissing(OpenEOException):
    pass


class ModelNotFoundException(OpenEOException):
    pass


class DimensionNotAvailable(OpenEOException):
    pass


class OverlapResolverMissing(OpenEOException):
    pass


class QuantilesParameterMissing(OpenEOException):
    pass


class QuantilesParameterConflict(OpenEOException):
    pass


class DimensionMissing(OpenEOException):
    pass


class BandFilterParameterMissing(OpenEOException):
    pass


class NoDataAvailable(OpenEOException):
    pass


class TemporalExtentEmpty(OpenEOException):
    pass


class DimensionAmbiguous(OpenEOException):
    pass


class NirBandAmbiguous(OpenEOException):
    pass


class RedBandAmbiguous(OpenEOException):
    pass


class BandExists(OpenEOException):
    pass


class DimensionMismatch(OpenEOException):
    pass


class LabelMismatch(OpenEOException):
    pass


class KernelDimensionsUneven(OpenEOException):
    pass


class MinMaxSwapped(OpenEOException):
    pass


class UnitMismatch(OpenEOException):
    pass
