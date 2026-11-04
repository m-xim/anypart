from __future__ import annotations

import dataclasses
import inspect
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type, TypeVar


class MultipartError(ValueError):
    """Base multipart error."""


class BoundaryNotFoundError(MultipartError):
    """Raised when multipart boundary is missing in content type."""


class InvalidMultipartError(MultipartError):
    """Raised when multipart payload cannot be parsed."""


class MissingPartError(MultipartError):
    """Raised when required model field is missing in parsed parts."""


@dataclass(frozen=True)
class Part:
    name: str
    data: bytes
    filename: Optional[str] = None
    content_type: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)


T = TypeVar("T")
PartDecoder = Callable[[Part], Any]
PartEncoder = Callable[[str, Any], Part]


def _quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        value = value[1:-1]
        value = value.replace('\\"', '"').replace("\\\\", "\\")
    return value


def _parse_params(value: str) -> tuple[str, Dict[str, str]]:
    segments = [segment.strip() for segment in value.split(";") if segment.strip()]
    if not segments:
        raise InvalidMultipartError("header parameter value is empty")

    base = segments[0].lower()
    params: Dict[str, str] = {}
    for segment in segments[1:]:
        if "=" not in segment:
            continue
        key, raw = segment.split("=", 1)
        params[key.strip().lower()] = _unquote(raw.strip())
    return base, params


def _extract_boundary(content_type: str) -> str:
    base, params = _parse_params(content_type)
    if base != "multipart/form-data":
        raise BoundaryNotFoundError("content type must be multipart/form-data")
    boundary = params.get("boundary")
    if not boundary:
        raise BoundaryNotFoundError("missing boundary in content type")
    return boundary


def _serialize_part(part: Part) -> bytes:
    disposition = f"form-data; name={_quote(part.name)}"
    if part.filename is not None:
        disposition += f"; filename={_quote(part.filename)}"

    headers = {"Content-Disposition": disposition}
    if part.content_type:
        headers["Content-Type"] = part.content_type
    for key, value in part.headers.items():
        headers[key] = value

    header_bytes = b"\r\n".join(f"{key}: {value}".encode("utf-8") for key, value in headers.items())
    return header_bytes + b"\r\n\r\n" + part.data


def build_multipart(parts: Sequence[Part], boundary: Optional[str] = None) -> tuple[bytes, str]:
    multipart_boundary = boundary or f"anypart-{uuid.uuid4().hex}"
    encoded_boundary = multipart_boundary.encode("utf-8")

    body_chunks: List[bytes] = []
    for part in parts:
        body_chunks.append(b"--" + encoded_boundary + b"\r\n" + _serialize_part(part) + b"\r\n")
    body_chunks.append(b"--" + encoded_boundary + b"--\r\n")

    body = b"".join(body_chunks)
    content_type = f"multipart/form-data; boundary={multipart_boundary}"
    return body, content_type


def _parse_headers(header_blob: bytes) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for line in header_blob.split(b"\r\n"):
        if not line:
            continue
        if b":" not in line:
            raise InvalidMultipartError("malformed header line")
        key, value = line.split(b":", 1)
        headers[key.decode("utf-8").strip()] = value.decode("utf-8").strip()
    return headers


def parse_multipart(body: bytes, content_type: str) -> List[Part]:
    boundary = _extract_boundary(content_type)
    delimiter = b"--" + boundary.encode("utf-8")
    if delimiter not in body:
        raise InvalidMultipartError("boundary not found in payload")

    sections = body.split(delimiter)
    parts: List[Part] = []
    for section in sections[1:]:
        if section.startswith(b"--"):
            break
        if section.startswith(b"\r\n"):
            section = section[2:]
        if section.endswith(b"\r\n"):
            section = section[:-2]
        if not section:
            continue

        if b"\r\n\r\n" not in section:
            raise InvalidMultipartError("part does not contain header separator")
        raw_headers, data = section.split(b"\r\n\r\n", 1)
        headers = _parse_headers(raw_headers)

        raw_disposition = headers.get("Content-Disposition")
        if not raw_disposition:
            raise InvalidMultipartError("part missing Content-Disposition")

        disposition, params = _parse_params(raw_disposition)
        if disposition != "form-data":
            raise InvalidMultipartError("unsupported disposition")

        name = params.get("name")
        if not name:
            raise InvalidMultipartError("part name is missing")

        parts.append(
            Part(
                name=name,
                filename=params.get("filename"),
                content_type=headers.get("Content-Type"),
                data=data,
                headers={
                    key: value
                    for key, value in headers.items()
                    if key not in {"Content-Disposition", "Content-Type"}
                },
            )
        )

    return parts


def _default_decode_part(part: Part) -> Any:
    if part.content_type and part.content_type.lower().startswith("text/"):
        return part.data.decode("utf-8")
    return part.data


def _default_encode_value(name: str, value: Any) -> Part:
    if isinstance(value, Part):
        return value
    if isinstance(value, bytes):
        return Part(name=name, data=value, content_type="application/octet-stream")
    if isinstance(value, str):
        return Part(name=name, data=value.encode("utf-8"), content_type="text/plain; charset=utf-8")
    return Part(name=name, data=str(value).encode("utf-8"), content_type="text/plain; charset=utf-8")


def _normalize_model_dump(model: Any) -> Dict[str, Any]:
    if dataclasses.is_dataclass(model):
        return dataclasses.asdict(model)
    if isinstance(model, dict):
        return dict(model)
    if hasattr(model, "__dict__"):
        return dict(vars(model))
    raise TypeError("model cannot be converted to a mapping")


def model_to_parts(model: Any, *, retort: Any = None, part_encoder: Optional[PartEncoder] = None) -> List[Part]:
    encoder = part_encoder or _default_encode_value
    payload = retort.dump(model) if retort is not None else _normalize_model_dump(model)
    return [encoder(name, value) for name, value in payload.items()]


def parts_to_model(
    parts: Sequence[Part],
    model_type: Type[T],
    *,
    retort: Any = None,
    part_decoder: Optional[PartDecoder] = None,
) -> T:
    decoder = part_decoder or _default_decode_part

    data: Dict[str, Any] = {}
    for part in parts:
        data[part.name] = decoder(part)

    if retort is not None:
        return retort.load(data, model_type)

    if dataclasses.is_dataclass(model_type):
        missing = [field.name for field in dataclasses.fields(model_type) if field.init and field.name not in data]
        if missing:
            raise MissingPartError(f"missing multipart fields: {', '.join(missing)}")
    return model_type(**data)


def build_model_multipart(
    model: Any,
    *,
    boundary: Optional[str] = None,
    retort: Any = None,
    part_encoder: Optional[PartEncoder] = None,
) -> tuple[bytes, str]:
    return build_multipart(model_to_parts(model, retort=retort, part_encoder=part_encoder), boundary=boundary)


def parse_model_multipart(
    body: bytes,
    content_type: str,
    model_type: Type[T],
    *,
    retort: Any = None,
    part_decoder: Optional[PartDecoder] = None,
) -> T:
    return parts_to_model(parse_multipart(body, content_type), model_type, retort=retort, part_decoder=part_decoder)


async def parse_multipart_async(body_or_reader: Any, content_type: str) -> List[Part]:
    if isinstance(body_or_reader, (bytes, bytearray, memoryview)):
        payload = bytes(body_or_reader)
    else:
        reader = getattr(body_or_reader, "read", None)
        if reader is None or not callable(reader):
            raise TypeError("body_or_reader must be bytes-like or have a read() method")
        read_result = reader()
        payload = await read_result if inspect.isawaitable(read_result) else read_result
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("read() must return bytes-like payload")
        payload = bytes(payload)

    return parse_multipart(payload, content_type)


async def build_multipart_async(parts: Sequence[Part], boundary: Optional[str] = None) -> tuple[bytes, str]:
    return build_multipart(parts, boundary=boundary)


async def parse_model_multipart_async(
    body_or_reader: Any,
    content_type: str,
    model_type: Type[T],
    *,
    retort: Any = None,
    part_decoder: Optional[PartDecoder] = None,
) -> T:
    parts = await parse_multipart_async(body_or_reader, content_type)
    return parts_to_model(parts, model_type, retort=retort, part_decoder=part_decoder)


async def build_model_multipart_async(
    model: Any,
    *,
    boundary: Optional[str] = None,
    retort: Any = None,
    part_encoder: Optional[PartEncoder] = None,
) -> tuple[bytes, str]:
    return build_model_multipart(model, boundary=boundary, retort=retort, part_encoder=part_encoder)
