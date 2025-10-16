"""
Download initial plant dataset from iNaturalist API
Targets 10 common houseplant species with 100 images each

Author: Abhinav M
Dependencies: requests, tqdm
"""

import requests
import json
import sys
import time
from pathlib import Path
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.utils.logger import setup_logger
    logger = setup_logger("DatasetDownloader")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DatasetDownloader")

# Target species list (scientific name, common name)
SPECIES_LIST = [
    ("Monstera deliciosa", "Swiss Cheese Plant"),
    ("Epipremnum aureum", "Pothos"),
    ("Sansevieria trifasciata", "Snake Plant"),
    ("Chlorophytum comosum", "Spider Plant"),
    ("Spathiphyllum wallisii", "Peace Lily"),
    ("Ficus lyrata", "Fiddle Leaf Fig"),
    ("Aloe vera", "Aloe Vera"),
    ("Crassula ovata", "Jade Plant"),
    ("Hedera helix", "English Ivy"),
    ("Ficus elastica", "Rubber Plant"),
]


class iNaturalistDownloader:
    """Download research-grade plant images from iNaturalist"""
    
    def __init__(self, output_dir="data/raw"):
        self.base_url = "https://api.inaturalist.org/v1"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.stats = {
            "total_downloaded": 0,
            "total_failed": 0,
            "species_results": {}
        }
    
    def search_taxon(self, scientific_name):
        """
        Find taxon ID for a species
        
        Args:
            scientific_name: Scientific name of plant species
            
        Returns:
            tuple: (taxon_id, taxon_name) or (None, None) if not found
        """
        url = f"{self.base_url}/taxa"
        params = {
            "q": scientific_name,
            "rank": "species",
            "is_active": "true"
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            results = response.json()["results"]
            
            if results:
                taxon_id = results[0]["id"]
                taxon_name = results[0]["name"]
                logger.info(f"Found taxon: {taxon_name} (ID: {taxon_id})")
                return taxon_id, taxon_name
            else:
                logger.warning(f"No taxon found for: {scientific_name}")
                return None, None
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Taxon search failed for {scientific_name}: {e}")
            return None, None
        except Exception as e:
            logger.error(f"Unexpected error searching for {scientific_name}: {e}")
            return None, None
    
    def download_observations(self, taxon_id, taxon_name, max_images=100):
        """
        Download observation images for a taxon
        
        Args:
            taxon_id: iNaturalist taxon ID
            taxon_name: Name of the taxon
            max_images: Maximum number of images to download
            
        Returns:
            int: Number of images successfully downloaded
        """
        url = f"{self.base_url}/observations"
        params = {
            "taxon_id": taxon_id,
            "photos": "true",
            "quality_grade": "research",  # Only expert-verified
            "per_page": min(max_images, 200),  # API limit is 200
            "order": "votes",  # Get highest quality images first
        }
        
        try:
            logger.info(f"Fetching observations for {taxon_name}...")
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            observations = response.json()["results"]
            
            if not observations:
                logger.warning(f"No observations found for {taxon_name}")
                return 0
            
            # Create species directory (safe filename)
            safe_name = taxon_name.replace(" ", "_").replace("/", "-").replace("'", "")
            species_dir = self.output_dir / safe_name
            species_dir.mkdir(exist_ok=True)
            
            logger.info(f"Found {len(observations)} observations for {taxon_name}")
            logger.info(f"Downloading to: {species_dir}")
            
            downloaded = 0
            failed = 0
            
            # Download with progress bar
            pbar = tqdm(total=min(max_images, len(observations)), 
                       desc=safe_name, 
                       unit="img")
            
            for obs in observations:
                if downloaded >= max_images:
                    break
                
                if not obs.get("photos"):
                    continue
                
                # Get medium resolution image URL
                photo = obs["photos"][0]
                photo_url = photo["url"].replace("square", "medium")
                obs_id = obs["id"]
                
                # Create filename
                img_path = species_dir / f"{safe_name}_{obs_id}.jpg"
                
                # Skip if already downloaded
                if img_path.exists():
                    logger.debug(f"Skipping existing file: {img_path.name}")
                    downloaded += 1
                    pbar.update(1)
                    continue
                
                # Download image
                try:
                    img_response = self.session.get(photo_url, timeout=10)
                    img_response.raise_for_status()
                    
                    # Save image
                    with open(img_path, 'wb') as f:
                        f.write(img_response.content)
                    
                    downloaded += 1
                    pbar.update(1)
                    
                    # Rate limiting - be respectful to API
                    time.sleep(0.5)
                
                except requests.exceptions.RequestException as e:
                    logger.debug(f"Failed to download {photo_url}: {e}")
                    failed += 1
                    continue
                except Exception as e:
                    logger.debug(f"Unexpected error downloading {photo_url}: {e}")
                    failed += 1
                    continue
            
            pbar.close()
            
            logger.info(f"✅ {taxon_name}: Downloaded {downloaded} images ({failed} failed)")
            return downloaded
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch observations for {taxon_name}: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error processing {taxon_name}: {e}")
            return 0
    
    def download_species_list(self, species_list, images_per_species=100):
        """
        Download all species in the provided list
        
        Args:
            species_list: List of tuples (scientific_name, common_name)
            images_per_species: Number of images to download per species
            
        Returns:
            dict: Download statistics
        """
        logger.info("=" * 70)
        logger.info("🌿 PlantaeClassScanner Dataset Downloader")
        logger.info("=" * 70)
        logger.info(f"Target: {len(species_list)} species")
        logger.info(f"Images per species: {images_per_species}")
        logger.info(f"Total target images: {len(species_list) * images_per_species}")
        logger.info(f"Output directory: {self.output_dir.absolute()}")
        logger.info("")
        
        for idx, (scientific_name, common_name) in enumerate(species_list, 1):
            logger.info("=" * 70)
            logger.info(f"[{idx}/{len(species_list)}] Processing: {common_name}")
            logger.info(f"Scientific name: {scientific_name}")
            logger.info("=" * 70)
            
            # Find taxon
            taxon_id, taxon_name = self.search_taxon(scientific_name)
            
            if not taxon_id:
                self.stats["species_results"][common_name] = {
                    "status": "failed",
                    "count": 0,
                    "error": "Taxon not found"
                }
                logger.error(f"❌ Skipping {common_name} - taxon not found\n")
                continue
            
            # Download images
            count = self.download_observations(taxon_id, taxon_name, images_per_species)
            
            self.stats["total_downloaded"] += count
            self.stats["species_results"][common_name] = {
                "status": "success" if count > 0 else "failed",
                "count": count,
                "taxon_name": taxon_name
            }
            
            logger.info("")
            
            # Be nice to the API - pause between species
            if idx < len(species_list):
                time.sleep(2)
        
        # Print final summary
        self._print_summary()
        
        return self.stats
    
    def _print_summary(self):
        """Print download summary statistics"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 DOWNLOAD SUMMARY")
        logger.info("=" * 70)
        
        success_count = 0
        failed_count = 0
        
        for species, result in self.stats["species_results"].items():
            status_icon = "✅" if result["status"] == "success" else "❌"
            count = result["count"]
            
            if result["status"] == "success":
                success_count += 1
                logger.info(f"{status_icon} {species:30s} {count:4d} images")
            else:
                failed_count += 1
                error = result.get("error", "Unknown error")
                logger.error(f"{status_icon} {species:30s} FAILED - {error}")
        
        logger.info("=" * 70)
        logger.info(f"Total Images Downloaded: {self.stats['total_downloaded']}")
        logger.info(f"Successful Species: {success_count}/{len(self.stats['species_results'])}")
        logger.info(f"Failed Species: {failed_count}/{len(self.stats['species_results'])}")
        logger.info("=" * 70)
        
        if self.stats['total_downloaded'] > 0:
            logger.info("✅ Dataset download complete!")
            logger.info(f"📁 Location: {self.output_dir.absolute()}")
        else:
            logger.error("❌ No images were downloaded!")


def main():
    """Main execution"""
    downloader = iNaturalistDownloader()
    downloader.download_species_list(SPECIES_LIST, images_per_species=100)


if __name__ == "__main__":
    main()