import os
from dotenv import load_dotenv

import xarray as xr
from typing import Dict, Optional, Any

def get_raster2stac_class():
    from raster2stac import Raster2STAC
    return Raster2STAC

# Default provider
EURAC_RESEARCH_PROVIDER = {
    "name": "Eurac Research - Institute for Earth Observation",
    "url": "http://www.eurac.edu",
    "roles": ["processor"],
}

def raster2stac(
    data: xr.DataArray,
    item_id: str,
    collection_url: str,
    output_folder: str,
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
    context: Optional[Dict[str, Any]] = None,
) -> xr.DataArray:
    """
    OpenEO process to generate STAC metadata from a RasterCube using Raster2STAC,
    and return the original input unchanged to support chaining.
    """

    #aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
    #aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
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
        output_folder=output_folder,
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

    return data
