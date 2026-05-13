"""In-memory storage for download sessions and fitted SDV models.

Replace with S3 / a database for production deployments.
"""

sessions: dict[str, dict] = {}
sdv_models: dict[str, dict] = {}
