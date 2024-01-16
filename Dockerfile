# lightweight python
FROM python:3.11

RUN pip install --upgrade pip
# Copy local code to the container image.
WORKDIR /app
COPY . /app



# Install dependencies
RUN pip install -r requirements.txt
EXPOSE 80
# Run the streamlit on container startup
CMD ["streamlit", "run", "app.py"]