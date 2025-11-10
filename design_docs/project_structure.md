# Project Structure

This document outlines the overall architecture and file structure of the VidSeq application.

## Overall Project Structure

```
vidseq/                          # Project root
├── vidseq/                      # Python package (backend)
│   ├── __init__.py
│   ├── cli.py                   # CLI entry point
│   ├── server.py                 # FastAPI app instance
│   ├── database.py              # Database connection & setup
│   ├── api/                      # API routes
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── projects.py      # Project endpoints
│   │       ├── videos.py         # Video endpoints
│   │       └── ...
│   ├── models/                   # SQLAlchemy models
│   │   ├── __init__.py
│   │   ├── project.py
│   │   ├── video.py
│   │   └── ...
│   ├── schemas/                  # Pydantic schemas (request/response)
│   │   ├── __init__.py
│   │   ├── project.py
│   │   └── ...
│   ├── services/                # Business logic
│   │   ├── __init__.py
│   │   ├── project_service.py
│   │   └── ...
│   └── frontend_dist/            # Built frontend files (included in package)
│       ├── index.html
│       └── assets/
│           └── ...
│
├── frontend/                     # Vue 3 frontend (source)
│   ├── src/
│   │   ├── components/           # Reusable Vue components
│   │   │   ├── ProjectCard.vue
│   │   │   ├── VideoList.vue
│   │   │   └── ...
│   │   ├── views/                # Page-level components (routes)
│   │   │   ├── HomeView.vue      # Start screen
│   │   │   ├── NewProjectView.vue
│   │   │   ├── ProjectView.vue
│   │   │   └── ...
│   │   ├── stores/               # Pinia stores (state management)
│   │   │   ├── project.ts
│   │   │   └── ...
│   │   ├── composables/          # Reusable composition functions
│   │   │   ├── useApi.ts
│   │   │   └── useWebSocket.ts
│   │   ├── services/             # API client functions
│   │   │   └── api.ts
│   │   ├── router/               # Vue Router configuration
│   │   │   └── index.ts
│   │   ├── App.vue               # Root component
│   │   ├── main.ts               # Entry point
│   │   └── assets/               # CSS, images, fonts
│   │       └── style.css
│   ├── index.html                # HTML entry point
│   ├── vite.config.ts            # Vite configuration
│   ├── tsconfig.json             # TypeScript config
│   └── package.json
│
├── design_docs/                  # Design documentation
├── pyproject.toml                # Python package config
├── README.md
└── .gitignore
```