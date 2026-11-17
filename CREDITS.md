# AresAtlas — Credits & Acknowledgments

AresAtlas is an independent planetary robotics and computer vision research platform exploring semantic segmentation, predictive uncertainty, and decision-aware route planning on Mars.

---

## Scientific Dataset & Primary Citations

The empirical evaluations in AresAtlas are conducted on the **AI4Mars** dataset released by NASA Jet Propulsion Laboratory (JPL):

```bibtex
@inproceedings{swan2021ai4mars,
  title     = {AI4MARS: A Dataset for Terrain-Aware Autonomous Driving on Mars},
  author    = {Swan, Reid and Atha, David and Ono, Masahiro and Sorice, Curtis and 
               Cuviello, Matthew and Lu, Yang and Person, Keri and 
               Villalpando, Christopher and Rossi, Brandon and 
               Gaines, Daniel and others},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  pages     = {1982--1991},
  year      = {2021}
}
```

- **Curiosity Rover Imagery:** Captured by the Navigation Cameras (Navcam) aboard the Mars Science Laboratory (*Curiosity*) rover operating in Gale Crater, Mars.
- **Data Access:** Published via the NASA Open Data Portal and mirrored on Hugging Face Datasets (`hassanjbara/AI4MARS`).
- **Expert Labels:** Test partition evaluations use the consensus gold-standard expert labels (`test_min1`).

---

## Open-Source Software & Tooling

AresAtlas is built upon and inspired by the following open-source libraries and frameworks:

- **Deep Learning & Computer Vision:**
  - [PyTorch](https://pytorch.org/) (Linux Foundation / Meta AI) — Tensor computation, autograd, and Apple Silicon Metal Performance Shaders (MPS) acceleration.
  - [torchvision](https://github.com/pytorch/vision) — Computer vision utilities and model primitives.
- **Backend Infrastructure:**
  - [FastAPI](https://fastapi.tiangolo.com/) (Sebastián Ramírez) — High-performance async REST API framework.
  - [Uvicorn](https://www.uvicorn.org/) — ASGI web server implementation.
  - [Pydantic](https://docs.pydantic.dev/) — Data validation and settings management.
- **Numerical Computing & Graph Algorithms:**
  - [NumPy](https://numpy.org/) — Multidimensional array operations and raster grid manipulation.
  - [SciPy](https://scipy.org/) — Spatial algorithms, morphological filters, and statistical utilities.
  - Priority Queue Graph Solvers — Dijkstra and A* pathfinding heuristics.
- **Testing & Quality Assurance:**
  - [pytest](https://docs.pytest.org/) — Automated unit, regression, and property test execution.

---

## Typography & Visual Assets

- **Typography:**
  - *Cormorant Garamond* — Designed by Christian Thalmann (Catharsis Fonts), licensed under the SIL Open Font License.
  - *Inter* — Designed by Rasmus Andersson, licensed under the SIL Open Font License.
- **Imagery:**
  - Mars surface and landscape imagery courtesy of NASA / JPL-Caltech / MSSS.
  - Environmental hero renders are original illustrative artwork designed for aesthetic depth without obscuring genuine rover telemetry.

---

## Institutional Disclaimer

AresAtlas is an independent research project. It is **not** endorsed by, affiliated with, or sponsored by NASA, the Jet Propulsion Laboratory (JPL), or the California Institute of Technology (Caltech). Any opinions, findings, and conclusions expressed in this repository are those of the authors.
