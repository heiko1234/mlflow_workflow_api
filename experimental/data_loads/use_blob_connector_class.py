




from experimental.data_loads.blob_connector_class import BlobStorageConnector


BlobConnector = BlobStorageConnector(
    storage_account_name="devstoreaccount1",
    container_name="chemical-data",
    local_run=True
    )


BlobConnector.list_all_files(subcontainer="chemical-data", files_with=".parquet")













