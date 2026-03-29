"""Add OBB columns to existing project databases.

Usage: uv run python scripts/migrate_add_obb_columns.py /path/to/project
"""
import sqlite3
import sys
from pathlib import Path


OBB_COLUMNS = {
    "obb_x1": "FLOAT",
    "obb_y1": "FLOAT",
    "obb_x2": "FLOAT",
    "obb_y2": "FLOAT",
    "obb_x3": "FLOAT",
    "obb_y3": "FLOAT",
    "obb_x4": "FLOAT",
    "obb_y4": "FLOAT",
    "obb_score": "FLOAT NOT NULL DEFAULT -1.0",
}


def find_db(project_path: Path) -> Path | None:
    """Find vidseq.db under the project path."""
    db = project_path / "vidseq.db"
    if db.exists():
        return db
    for db in project_path.rglob("vidseq.db"):
        return db
    return None


def migrate(db_path: Path) -> None:
    """Add OBB columns to frame_data table."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(frame_data)")
    existing = {row[1] for row in cursor.fetchall()}

    added = 0
    for col_name, col_type in OBB_COLUMNS.items():
        if col_name not in existing:
            cursor.execute(f"ALTER TABLE frame_data ADD COLUMN {col_name} {col_type}")
            added += 1
            print(f"  Added column: {col_name}")

    conn.commit()
    conn.close()

    if added == 0:
        print("  All OBB columns already exist, nothing to do.")
    else:
        print(f"  Added {added} columns.")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} /path/to/project")
        sys.exit(1)

    project_path = Path(sys.argv[1])
    if not project_path.exists():
        print(f"Error: {project_path} does not exist")
        sys.exit(1)

    db_path = find_db(project_path)
    if db_path is None:
        print(f"Error: No vidseq.db found under {project_path}")
        sys.exit(1)

    print(f"Migrating: {db_path}")
    migrate(db_path)
    print("Done.")


if __name__ == "__main__":
    main()
