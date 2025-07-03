import os
import json
import requests
from dotenv import load_dotenv

import xarray as xr
from typing import Dict, Optional, Any

def get_raster2stac_class():
    from raster2stac import Raster2STAC
    return Raster2STAC

EURAC_RESEARCH_PROVIDER = {
    "name": "Eurac Research - Institute for Earth Observation",
    "url": "http://www.eurac.edu",
    "roles": ["processor"],
}

def raster2stac(
    data: xr.DataArray,
    item_id: str,
    collection_url: str,
    description: str,
    write_collection_assets: bool = True,
    license: str = "Apache-2.0",
    keywords: Optional[list] = None,
    providers: Optional[list] = None,
    sci_citation: Optional[str] = None,
    sci_doi: Optional[str] = None,
    links: Optional[list] = None,
    s3_upload: bool = False,
    s3_endpoint_url: Optional[str] = None,
    bucket_name: Optional[str] = None,
    bucket_file_prefix: Optional[str] = "",
    post_to_stac: bool = False,
    context: Optional[Dict[str, Any]] = None,
) -> xr.DataArray:
    """
    OpenEO process to generate STAC metadata from a RasterCube using Raster2STAC,
    and optionally post it to a remote STAC API. Uses item_id as output folder and collection ID.
    """

    load_dotenv()
    aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if s3_upload and (not aws_key or not aws_secret):
        raise RuntimeError("Missing AWS credentials in environment for S3 upload.")

    raster2stac_args = dict(
        data=data,
        collection_id=item_id,
        description=description,
        license=license,
        keywords=keywords or [],
        collection_url=collection_url,
        output_folder=item_id,  # Using item_id as output_folder
        providers=providers or [EURAC_RESEARCH_PROVIDER],
        sci_citation=sci_citation,
        sci_doi=sci_doi,
        links=links or [],
        s3_upload=s3_upload,
        write_collection_assets=write_collection_assets,
    )

    if s3_upload:
        if not s3_endpoint_url or not bucket_name:
            raise ValueError("Missing required S3 settings.")
        raster2stac_args.update(
            aws_access_key=aws_key,
            aws_secret_key=aws_secret,
            s3_endpoint_url=s3_endpoint_url,
            bucket_name=bucket_name,
            bucket_file_prefix=bucket_file_prefix or "",
        )

    Raster2STAC = get_raster2stac_class()
    stac = Raster2STAC(**raster2stac_args)
    stac.generate_zarr_stac(item_id=item_id)

    # Optional STAC POST
    if post_to_stac:
        base_path = os.path.join(item_id)
        collection_json_path = os.path.join(base_path, f"{item_id}.json")
        items_csv_path = os.path.join(base_path, "inline_items.csv")

        if not os.path.exists(collection_json_path):
            raise FileNotFoundError(f"Collection file missing at {collection_json_path}")
        if not os.path.exists(items_csv_path):
            raise FileNotFoundError(f"Items file missing at {items_csv_path}")

        requests.delete(f"{collection_url}{item_id}")

        with open(collection_json_path, "r") as f:
            collection = json.load(f)
        response = requests.post(collection_url, json=collection)
        if response.status_code >= 400:
            raise RuntimeError(f"Collection POST failed: {response.text}")

        with open(items_csv_path, "r") as f:
            for line in f:
                item = json.loads(line)
                item_url = f"{collection_url}{item_id}/items"
                resp = requests.post(item_url, json=item)
                if resp.status_code >= 400:
                    raise RuntimeError(f"Item POST failed: {resp.text}")

    return data
