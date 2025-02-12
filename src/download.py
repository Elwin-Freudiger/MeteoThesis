import os
import asyncio
import aiohttp
import logging
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "data/raw"
MAX_CONCURRENT_DOWNLOADS = 2

ios.makedirs(DATA_DIR, exist_ok=True)

class OpenDataAPI:
    def __init__(self, api_token: str):
        self.base_url = "https://api.dataplatform.knmi.nl/open-data/v1"
        self.headers = {"Authorization": api_token}

    async def __get_data(self, session, url, params=None):
        async with session.get(url, headers=self.headers, params=params) as response:
            return await response.json()

    async def get_file_url(self, session, dataset_name, dataset_version, file_name):
        url = f"{self.base_url}/datasets/{dataset_name}/versions/{dataset_version}/files/{file_name}/url"
        return await self.__get_data(session, url)

async def download_file(session, file_url, filename):
    file_path = os.path.join(DATA_DIR, filename)
    try:
        async with session.get(file_url) as response:
            if response.status != 200:
                logger.error(f"Failed to download {filename}: HTTP {response.status}")
                return False
            
            with open(file_path, "wb") as f:
                f.write(await response.read())
            
            logger.info(f"Successfully downloaded {filename}")
            return True
    except Exception as e:
        logger.error(f"Error downloading {filename}: {e}")
        return False

async def download_files(api, dataset_name, dataset_version, file_list):
    connector = aiohttp.TCPConnector(limit_per_host=MAX_CONCURRENT_DOWNLOADS)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for filename in file_list:
            response = await api.get_file_url(session, dataset_name, dataset_version, filename)
            if "temporaryDownloadUrl" in response:
                file_url = response["temporaryDownloadUrl"]
                tasks.append(download_file(session, file_url, filename))
            else:
                logger.error(f"Failed to get download URL for {filename}")
        
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Downloading files"):
            await f

async def main():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    dataset_name = "RAD_OPERA_HOURLY_RAINFALL_ACCUMULATION_EURADCLIM"
    dataset_version = "2.0"
    
    api = OpenDataAPI(api_token=api_key)
    
    # Generate filenames for the period Jan 2019 - Dec 2021
    file_list = [
        f"RAD_OPERA_HOURLY_RAINFALL_ACCUMULATION_EURADCLIM_{year}{month:02d}_0002.zip"
        for year in range(2019, 2022) for month in range(1, 13)
    ]
    
    logger.info(f"Starting downloads for {len(file_list)} files...")
    await download_files(api, dataset_name, dataset_version, file_list)
    logger.info("All downloads complete.")

if __name__ == "__main__":
    asyncio.run(main())