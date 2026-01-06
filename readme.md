# Molecular Pipeline

## Environment Configuration

The application uses a single `.env` file for all environment variables. Set the `ENV` variable in the `.env` file to control the mode:

- **`ENV=development`** or **`ENV=dev`** - Development mode (enables file saving, debug logging)
- **`ENV=production`** or **`ENV=prod`** or unset - Production mode (default)

### Running the Server

**Development mode:**
Set `ENV=development` in your `.env` file, then:
```bash
uvicorn app:app --host 0.0.0.0 --port 7653
```

**Production mode:**
Set `ENV=production` in your `.env` file (or leave it unset), then:
```bash
uvicorn app:app --host 0.0.0.0 --port 7653
```

The application will automatically detect the mode from the `ENV` variable in your `.env` file.

## Docker Deployment

### Building and Running with Docker

**Using Docker Compose (Recommended):**
```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

**Using Docker directly:**
```bash
# Build the image
docker build -t molecular-pipeline .

# Run the container
docker run -d \
  --name molecular-pipeline \
  -p 7653:7653 \
  --env-file .env \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/key:/app/key:ro \
  molecular-pipeline
```

### Docker Requirements

- Docker and Docker Compose installed
- `.env` file with all required environment variables
- `key` file in project root (for SSH tunnel, if using database feature)

The Docker container includes:
- Python 3.11
- Google Chrome (for Selenium)
- ChromeDriver
- All Python dependencies
- NLTK data pre-downloaded
