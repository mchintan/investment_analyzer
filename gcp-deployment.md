# Deploying Investment Analyzer to Google Cloud Platform

This guide provides step-by-step instructions for deploying the Investment Analyzer application to Google Cloud Platform using Docker and Cloud Run.

## Prerequisites

1. Google Cloud Platform account
2. Google Cloud SDK installed locally
3. Docker installed locally
4. Git repository with your Investment Analyzer code

## Deployment Steps

### 1. Build and Test Docker Image Locally

```bash
# Navigate to your project directory
cd /path/to/investment_analyzer

# Build the Docker image
docker build -t investment-analyzer:latest .

# Run the container locally to test
docker run -p 8501:8501 investment-analyzer:latest
```

Visit `http://localhost:8501` in your browser to confirm the application works correctly.

### 2. Configure Google Cloud Project

```bash
# Set your GCP project ID
export PROJECT_ID=your-gcp-project-id

# Configure Docker to use Google Cloud as a registry
gcloud auth configure-docker

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
```

### 3. Manual Deployment to Cloud Run

```bash
# Build and tag your Docker image
docker build -t gcr.io/$PROJECT_ID/investment-analyzer:latest .

# Push the image to Google Container Registry
docker push gcr.io/$PROJECT_ID/investment-analyzer:latest

# Deploy the image to Cloud Run
gcloud run deploy investment-analyzer \
  --image gcr.io/$PROJECT_ID/investment-analyzer:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### 4. Continuous Deployment with Cloud Build

To set up continuous deployment:

1. Connect your GitHub repository to Cloud Build:
   ```bash
   gcloud builds triggers create github \
     --repo=your-github-username/investment_analyzer \
     --branch-pattern=main \
     --build-config=cloudbuild.yaml
   ```

2. Cloud Build will automatically deploy your application when you push to the main branch.

### 5. Access Your Deployed Application

After deployment, Cloud Run will provide a URL to access your application. You can also find this URL in the Cloud Run dashboard.

## Additional Configuration Options

### Environment Variables

If your application requires environment variables:

```yaml
# In cloudbuild.yaml, add these flags to the Cloud Run deploy command
- '--set-env-vars=VAR_NAME1=value1,VAR_NAME2=value2'
```

### Custom Domain Name

To use a custom domain:

1. Verify domain ownership in Google Cloud Console
2. Map the domain to your Cloud Run service:
   ```bash
   gcloud beta run domain-mappings create \
     --service investment-analyzer \
     --domain your-domain.com \
     --region us-central1
   ```

3. Update your DNS settings according to the provided instructions

### Cost Management

Cloud Run charges based on usage. To manage costs:

- Set memory limits in your deployment configuration
- Configure concurrency appropriately
- Consider setting CPU allocation to 1 or less for lower costs

## Troubleshooting

- **Application crashes:** Check Cloud Run logs for error messages
- **Performance issues:** Consider increasing memory allocation
- **Deployment failures:** Verify that your Dockerfile works locally first
- **Container startup issues:** Make sure the ENTRYPOINT command is correct

For more help, refer to the [Google Cloud Run documentation](https://cloud.google.com/run/docs).