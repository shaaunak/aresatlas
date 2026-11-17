# Backend integration contract

The standalone frontend assumes a research API available at `http://localhost:8000` by default. It is tolerant of missing endpoints and does not replace them with invented data.

- `GET /model/info`: JSON describing the trained model. Used to check model presence before prediction.
- `POST /predict`: `multipart/form-data` with a `file` field. The frontend currently understands a PNG visualization in `overlay_png_base64`, `overlay_base64`, or `overlay` returned in JSON. Adapt the front end and server contract together if your API differs.
- `GET /metrics`: JSON containing numeric fields such as `mean_iou`/`miou`, `sand_recall`, `big_rock_recall`, `inference_ms`, either directly or within `metrics`.
- `GET /experiments`: JSON array or `{ "experiments": [ { "id": "...", "name": "...", "status": "..." } ] }`.

Other modules (route computation and uncertainty study) presently run as transparent, deterministic, synthetic demonstrations entirely in the browser; they are not sent to an ML backend.

To use the UI as a genuine research results viewer, connect each panel to exact verified artifacts and state provenance. Do not silently repurpose the synthetic values as measured AI4Mars results.
