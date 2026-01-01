# Molecular Pipeline

## Environment Configuration

The application supports separate configuration files for development and production:

- **`.env`** - Production environment variables (default)
- **`.env.dev`** - Development environment variables

### Running the Server

**Development (uses `.env.dev`):**
```bash
ENV=dev uvicorn app:app --host 0.0.0.0 --port 5000
```

**Production (uses `.env`):**
```bash
uvicorn app:app --host 0.0.0.0 --port 5000
```

The application will automatically load the appropriate `.env` file based on the `ENV` environment variable.
