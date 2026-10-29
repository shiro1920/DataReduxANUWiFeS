import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor
import os


class BaseDownloader:
    def __init__(self, username, password):
        self.username = username
        self.password = password

        self.session = requests.Session()
        self.cache = {}  # cache: url → (soup, date_list)

    def fetch_soup(self, url):
        if url in self.cache:
            return self.cache[url]

        resp = self.session.get(url, auth=(self.username, self.password))
        soup = BeautifulSoup(resp.text, "html.parser")

        dates = []
        for row in soup.find_all("tr"):
            t = row.find("td", class_="indexcollastmod")
            if t:
                date_str = t.text.strip().split(" ")[0]
                dates.append(date_str)

        self.cache[url] = (soup, dates)
        return soup, dates

    def list_directories(self, url, depth=0, max_depth=10, url_list=None, date_list=None):
        if url_list is None: url_list = []
        if date_list is None: date_list = []

        soup, dates = self.fetch_soup(url)
        date_list.extend(dates)

        dirs = []
        for link in soup.find_all("a"):
            href = link.get("href")
            if href and href.endswith("/") and href not in ("../", "/"):
                full_url = urljoin(url, href)
                url_list.append(full_url)
                dirs.append(full_url)

        if depth < max_depth:
            for d in dirs:
                self.list_directories(d, depth + 1, max_depth, url_list, date_list)

        return url_list, date_list

    def list_files_recursive(self, url, collector=None, depth=0, max_depth=10):
        if collector is None: collector = []
        if depth > max_depth:
            return collector

        soup, _ = self.fetch_soup(url)

        for link in soup.find_all("a"):
            href = link.get("href")
            if not href or href.startswith("?") or href.startswith("/"):
                continue

            full_url = urljoin(url, href)

            if href.endswith("/"):
                self.list_files_recursive(full_url, collector, depth + 1)
            else:
                collector.append(full_url)

        return collector

    def run_scripts(self, script_paths, max_workers=6):
        if len(script_paths) == 0:
            print("No scripts to run.")
            return

        print(f"Running {len(script_paths)} scripts ...")

        def run_script(s):
            print(f"▶ Running: {s}")
            subprocess.run(["bash", str(s)], check=True)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(run_script, script_paths)

        print("Done running scripts.")


# ===========================================
# Calibration Data Downloader

class WiFeSCalibDownload(BaseDownloader):

    def __init__(self, username, password, obs_month, grisms, obs_date,
                 data_download_path, root_url="https://muggy.anu.edu.au/aocsarchive/content/Common/"):
        super().__init__(username, password)

        self.obs_month = obs_month
        self.grisms = grisms
        self.obs_date = obs_date
        self.root_url = root_url
        self.data_download_path = Path(data_download_path)

        self.cal_url = root_url + obs_month + '/'

        self.url_all = []
        self.date_all = []

        self.cal_keys = ['ARC', 'BIAS', 'DARK', 'FLAT', 'SKYFLAT', 'STANDARD', 'WIRE']
        self.calurl_dict = {}
        self.cal_files = {}

    def classify_calibration_urls(self):
        self.calurl_dict = {}

        biasA = []
        biasB = []
        dark = []
        grism_urls = []

        for u in self.url_all:

            # common bias/dark
            if "BIAS" in u and "/Obs" in u:
                if "Bias-A" in u:
                    biasA.append(u)
                elif "Bias-B" in u:
                    biasB.append(u)

            if "DARK" in u and "/Obs" in u and "Dark" in u:
                dark.append(u)

            # grism-specific
            if any(g in u for g in self.grisms) and "1x2-Full/Obs" in u:
                grism_urls.append(u)

        if biasA: self.calurl_dict["biasA"] = biasA
        if biasB: self.calurl_dict["biasB"] = biasB
        if dark: self.calurl_dict["dark"] = dark

        for key in self.cal_keys:
            if key in ["BIAS", "DARK"]:
                continue
            matched = [u for u in grism_urls if key in u]
            if matched:
                self.calurl_dict[key.lower()] = matched

    def extract_files(self):
        self.cal_files = {k: [] for k in self.calurl_dict}

        for key, urls in self.calurl_dict.items():
            for u in urls:
                files = self.list_files_recursive(u, [])
                for f in files:
                    if self.obs_date in f:
                        self.cal_files[key].append(f)

    def generate_or_detect_scripts(self):
        scripts = []

        for key, files in self.cal_files.items():
            cal_dir = self.data_download_path / key
            cal_dir.mkdir(parents=True, exist_ok=True)

            script_path = cal_dir / f"{self.obs_date}_{key}_download.sh"

            # If exists → do not regenerate
            if script_path.exists():
                scripts.append(script_path)
                continue

            with open(script_path, "w") as f:
                f.write("#!/bin/bash\n")
                f.write(f"cd \"{cal_dir.resolve()}\"\n\n")

                for url in files:
                    cmd = (
                        "wget -e robots=off -np -R 'index.html*' "
                        "--no-check-certificate --wait=1 -nc "
                        f"--user='{self.username}' --password='{self.password}' "
                        f"'{url}'\n"
                    )
                    f.write(cmd)

            os.chmod(script_path, 0o755)
            scripts.append(script_path)

        return scripts
    
    def run(self, download=False, max_workers=8):

        sh_files = list(Path(self.data_download_path).rglob("*.sh"))

        if not sh_files:
            print("No .sh file exists, start script generation.")
            self.url_all, self.date_all = self.list_directories(self.cal_url)
            self.classify_calibration_urls()
            self.extract_files()
            self.generate_or_detect_scripts()
            
            sh_files = list(Path(self.data_download_path).rglob("*.sh"))
            print("Scripts are under:", self.data_download_path)

        if download:
            if sh_files:
                print(f"Executing {len(sh_files)} scripts.")
                self.run_scripts(sh_files, max_workers)
            else:
                print("Warning: No scripts found to execute.")
    
# ===========================================
# Science Data Downloader
class WiFeSSciDownload(BaseDownloader):

    def __init__(self, username, password, semester, PID, obj_name,
                 data_download_path, root_url="https://muggy.anu.edu.au/aocsarchive/content/"):
        super().__init__(username, password)

        self.semester = semester
        self.PID = PID
        self.obj_name = obj_name if isinstance(obj_name, list) else [obj_name]
        self.root_url = root_url
        self.data_download_path = Path(data_download_path)

        self.sci_url = f"{root_url}{semester}/P-{PID}/"

        self.url_all = []
        self.date_all = []

        self.sciurl_dict = {}

    def extract_files(self):
        self.sciurl_dict = {obj: [] for obj in self.obj_name}

        for obj in self.obj_name:
            for idx, u in enumerate(self.url_all):
                if obj in u and "1x2-Full" in u:
                    date = self.date_all[idx] if idx < len(self.date_all) else "UNK"
                    self.sciurl_dict[obj].append([u, date])

    def generate_or_detect_scripts(self):
        scripts = []

        sci_path = self.data_download_path / "sci"
        sci_path.mkdir(parents=True, exist_ok=True)

        for obj, folder_list in self.sciurl_dict.items():
            script_path = sci_path / f"{obj}_download.sh"

            if script_path.exists():
                scripts.append(script_path)
                continue

            with open(script_path, "w") as f:
                f.write("#!/bin/bash\n")
                f.write(f"cd \"{sci_path.resolve()}\"\n\n")

                for url, date in folder_list:
                    cmd = (
                        "wget -e robots=off -np -r -R 'index.html*' "
                        "--no-check-certificate --wait=1 -nc "
                        f"--user='{self.username}' --password='{self.password}' "
                        f"'{url}'\n"
                    )
                    f.write(cmd)

            log_name = f"{obj}_raw_dataInfo.txt"
            log_path = sci_path / log_name
            with open(log_path, "w") as f:
                for file_info in folder_list:
                    file_folder = file_info[0]
                    OB_date = file_info[1]
                    f.write(f"{file_folder} {OB_date}\n")

            os.chmod(script_path, 0o755)
            scripts.append(script_path)

        return scripts

    def run(self, download=False, max_workers=8):

        sh_files = []
        for obj in self.obj_name:
            files = list(Path(self.data_download_path).rglob(f"*{obj}*.sh"))
            sh_files.extend(files)

        if not sh_files:
            print("No .sh files exist for objects, start script generation.")
            self.url_all, self.date_all = self.list_directories(self.sci_url)
            self.extract_files()
            self.generate_or_detect_scripts()
            
            sh_files = []
            for obj in self.obj_name:
                files = list(Path(self.data_download_path).rglob(f"*{obj}*.sh"))
                sh_files.extend(files)
                
            print(f"Scripts generated. Found {len(sh_files)} scripts under: {self.data_download_path}")

        if download and sh_files:
            unique_sh_files = list(set(sh_files))
            self.run_scripts(unique_sh_files, max_workers)


                        

