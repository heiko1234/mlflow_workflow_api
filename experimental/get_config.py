
def get_config(local_run: bool):
    
    
    load_dotenv()
    local_run = getenv("LOCAL_RUN", False)

    if local_run:
        try:
            config = read_configuration("./configuration/local_run.yaml")
        except Exception as e:
            print(e)
            config = read_configuration("./backend_service/configuration/production_run.yaml")
    else:
        try:
            config = read_configuration("configuration/production_run.yaml")
        except Exception as e:
            print(e)
            config = read_configuration("./backend_service/configuration/production_run.yaml")
        
    return config

