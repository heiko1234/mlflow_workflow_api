

from dotenv import load_dotenv

load_dotenv()


from upath import Upath


path = Upath("az.//chemical-data", account_name="devstoreaccount1", anon=False)

[p for p in path.iterdir()]
[str(p).split("/")[-1] for p in path.iterdir() if ".parquet" in str(p)]


















