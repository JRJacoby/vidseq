#!/usr/bin/env python3
"""Add associated video columns to existing project databases.

Usage: uv run python scripts/migrate_associated_videos.py /path/to/project
"""
import sqlite3
import sys
from pathlib import Path


def migrate(project_dir: Path) -> None:
    db_path = project_dir / "vidseq.db"
    if not db_path.exists():
        print(f"Error: {db_path} not found")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Check existing columns
    cursor.execute("PRAGMA table_info(videos)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    if "is_associated" not in existing_cols:
        cursor.execute(
            "ALTER TABLE videos ADD COLUMN is_associated INTEGER NOT NULL DEFAULT 0"
        )
        print("Added column: is_associated")
    else:
        print("Column is_associated already exists, skipping")

    if "associated_with_id" not in existing_cols:
        cursor.execute(
            "ALTER TABLE videos ADD COLUMN associated_with_id INTEGER DEFAULT NULL"
        )
        print("Added column: associated_with_id")
    else:
        print("Column associated_with_id already exists, skipping")

    conn.commit()
    conn.close()
    print("Migration complete.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run python scripts/migrate_associated_videos.py /path/to/project")
        sys.exit(1)
    migrate(Path(sys.argv[1]))
