# LMForge Setup Steps:

1. First, ensure you have requisite programs installed. You will need 

* Python (most of features have been tested on versions 3.10 and 3.11).
* MySQL 

2. Next, create a Python environment using the following command in the terminal (you should be located in the outer lmforge folder):

* For Windows:
- To create a virtual environment : python -m venv venv
- To activate virtual environment : .\venv\Scripts\activate

* For Linux / macOS:
- To create a virtual environment : python3 -m venv venv
- To activate virtual environment : source venv/bin/activate

3. Then, install dependencies. Most dependencies can be installed using the command below. However, torch with CUDA support must be configured according to your environment.
- pip install -r requirements.txt
- Visit the following site to retrieve the appropriate pip install for torch: https://pytorch.org/

4. Next, create .env file in the outer lmforge folder which should contain the following information:
    DATABASE_NAME=mydatabase 
    DATABASE_USER=myuser
    DATABASE_PASSWORD=mypassword
    DATABASE_HOST=localhost
    DATABASE_PORT=3306
    WANDB_API_KEY=yourKey
    HF_API_KEY=yourKey
    OPENAI_API_KEY=yourKey
- Be sure to change the values above to values for your system. 

5. Last, make sure your file location in your terminal is in the outer lmforge folder. Start your Python virtual environment with the first command below. Then, run the next three Python commands.
- .\venv\Scripts\activate
- python manage.py makemigrations
- python manage.py migrate
- python manage.py runserver

6. Whenever you want to start the applicatoin again, start your Python virtual environment and run the Python command below.
- .\venv\Scripts\activate
- python manage.py runserver 

7. If you want to setup qdrant vector db follow bellow steps: (Not required if you do not intend to use Vector db. )
- Podman must be installed on your system.
- If you don’t have it yet, follow the official installation guide:
- https://podman.io/getting-started/installation 
- Ensure your .env file includes these configuration values (adjust if needed):
    QDRANT_HOST=localhost
    QDRANT_PORT=6333
    QDRANT_LOG_FILE=application.log

8. Install Required Python Libraries:
-Add these dependencies to your Django project’s environment:
-pip install qdrant-client sentence-transformers
-If you’re using Hugging Face models for embedding or tokenizer operations, also install: 
-pip install transformers huggingface-hub

9. Run Qdrant with Podman
- Start a Qdrant container using the official image:
- podman run -d \
  --name qdrant \
  -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
- Explanation:
  - -d → Run in detached mode (background).
  - --name qdrant → Names the container for easy reference.
  - -p 6333:6333 → Maps Qdrant’s default API port.
  - -v qdrant_storage:/qdrant/storage → Persists your vector data even if the container restarts.
- qdrant/qdrant → Official Qdrant image from Docker Hub.

10. Verify Qdrant Is Running
- Check that the Qdrant container is active:
  podman ps
- You should see something like:
    CONTAINER ID  IMAGE                    COMMAND               CREATED         STATUS             PORTS                 NAMES
    abcd1234       docker.io/qdrant/qdrant  /usr/bin/qdrant ...   10 seconds ago  Up 10 seconds ago  0.0.0.0:6333->6333/tcp  qdrant
- Test connectivity by visiting:
- http://localhost:6333/dashboard

11. Configure Django to Use Qdrant
- Django app will automatically connect to Qdrant using the environment variables defined in .env.
- No additional manual setup is required if you followed the steps above.

12. Manage the Qdrant Container
- To stop Qdrant:
  podman stop qdrant
- To restart it later:
  podman start qdrant
- To remove the container (keeps data because of the named volume):
  podman rm qdrant

- Please use cmd for podman and qdrant setup and not VSCode.
