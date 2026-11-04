import asyncio
import dataclasses
import unittest

from anypart import (
    BoundaryNotFoundError,
    Part,
    build_model_multipart,
    build_multipart,
    build_multipart_async,
    model_to_parts,
    parse_model_multipart,
    parse_model_multipart_async,
    parse_multipart,
    parse_multipart_async,
)


@dataclasses.dataclass
class UploadModel:
    title: str
    payload: bytes


class StubRetort:
    def dump(self, model):
        return dataclasses.asdict(model)

    def load(self, data, model_type):
        return model_type(**data)


class AnyPartTests(unittest.TestCase):
    def test_sync_build_and_parse_roundtrip(self):
        parts = [
            Part(name="title", data=b"hello", content_type="text/plain; charset=utf-8"),
            Part(name="file", data=b"\x00\x01", filename="a.bin", content_type="application/octet-stream"),
        ]
        body, content_type = build_multipart(parts, boundary="fixed")

        parsed = parse_multipart(body, content_type)

        self.assertEqual(parsed, parts)

    def test_boundary_must_exist(self):
        with self.assertRaises(BoundaryNotFoundError):
            parse_multipart(b"", "multipart/form-data")

    def test_model_roundtrip_with_retort(self):
        model = UploadModel(title="hello", payload=b"\x10\x20")
        body, content_type = build_model_multipart(model, boundary="model", retort=StubRetort())

        restored = parse_model_multipart(body, content_type, UploadModel, retort=StubRetort())

        self.assertEqual(restored, model)

    def test_async_parse_build(self):
        parts = [Part(name="x", data=b"1", content_type="text/plain")]
        body, content_type = asyncio.run(build_multipart_async(parts, boundary="async"))

        class Reader:
            async def read(self):
                return body

        parsed = asyncio.run(parse_multipart_async(Reader(), content_type))
        self.assertEqual(parsed, parts)

    def test_async_model_roundtrip(self):
        model = UploadModel(title="world", payload=b"\x99")
        body, content_type = build_model_multipart(model, boundary="m2", retort=StubRetort())

        restored = asyncio.run(
            parse_model_multipart_async(body, content_type, UploadModel, retort=StubRetort())
        )

        self.assertEqual(restored, model)

    def test_model_to_parts_without_retort(self):
        model = UploadModel(title="a", payload=b"b")
        parts = model_to_parts(model)
        self.assertEqual({part.name for part in parts}, {"title", "payload"})


if __name__ == "__main__":
    unittest.main()
