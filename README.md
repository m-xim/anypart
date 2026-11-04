# anypart

Сборка и разборка `multipart/form-data` для sync и async сценариев.

Поддерживается:
- сборка и разборка списка частей (`Part`)
- сборка и разборка из модели (`model_to_parts` / `parts_to_model`)
- интеграция с `adaptix` через передачу `retort` (объект с `dump`/`load`)

## Пример (sync)

```python
from anypart import Part, build_multipart, parse_multipart

body, content_type = build_multipart([Part(name="a", data=b"1")])
parts = parse_multipart(body, content_type)
```

## Пример (async)

```python
from anypart import build_multipart_async, parse_multipart_async

body, content_type = await build_multipart_async([...])
parts = await parse_multipart_async(body, content_type)
```
