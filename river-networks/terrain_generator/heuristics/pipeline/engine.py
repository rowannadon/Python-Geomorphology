"""Core terrain heuristics engine integrated with the river-networks app."""

import os
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QImage

from .albedo import (
    compute_terrain_albedo_continuous,
    compute_terrain_albedo_physical,
    compute_terrain_albedo_rgb,
)
from .biome import BIOME_TABLE, classify_biomes_advanced
from .climate import (
    actual_evapotranspiration,
    directional_slope,
    latitude_degrees,
    potential_evapotranspiration,
    precipitation_lat_bands,
    precipitation_orographic_advanced,
    prevailing_wind,
    prevailing_wind_3cell,
    temperature_from_lat_elev,
)
from .curvature import compute_laplacian_curvature
from .flow import d8_flow_accumulation, d8_flow_direction
from .foliage import compute_foliage_color_rgb, compute_foliage_densities
from .ocean import compute_coastline_mask, compute_ocean_mask
from .slope_aspect import compute_normals, compute_slope_aspect
from .svf import compute_svf
from .terrain_io import load_heightmap, load_scalar_texture
from .tpi import compute_tpi
from .twi import compute_twi
from .util import distance_to_mask

def _percentile_clip(arr: np.ndarray, lo=2, hi=98):
    a = np.asarray(arr, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    plo, phi = np.percentile(a, [lo, hi])
    if phi <= plo: phi = plo + 1e-6
    a = (a - plo) / (phi - plo)
    return np.clip(a, 0, 1)

def scalar_to_qimage(arr: np.ndarray, lo=None, hi=None) -> QImage:
    a = np.asarray(arr, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if lo is None or hi is None:
        a = _percentile_clip(a)
    else:
        if hi <= lo: hi = lo + 1e-6
        a = np.clip((a - lo) / (hi - lo), 0, 1)
    img8 = (a * 255.0 + 0.5).astype(np.uint8)
    h, w = img8.shape
    qimg = QImage(img8.data, w, h, w, QImage.Format_Grayscale8)
    return qimg.copy()

def rgb_to_qimage(rgb: np.ndarray) -> QImage:
    # rgb expected uint8 (H,W,3)
    arr = np.asarray(rgb, dtype=np.uint8, order="C")
    h, w, _ = arr.shape
    qimg = QImage(arr.data, w, h, 3*w, QImage.Format_RGB888)
    return qimg.copy()

def normal_to_qimage(n: np.ndarray) -> QImage:
    n = np.nan_to_num(n, nan=0.0)
    n = (np.clip(n, -1, 1) * 0.5 + 0.5)
    out = (n * 255.0 + 0.5).astype(np.uint8)
    return rgb_to_qimage(out)

# -----------------------
# TerrainEngine (caching)
# -----------------------

class TerrainEngine(QObject):
    """Holds data in memory and computes only what’s needed."""
    progress = pyqtSignal(str, int)      # message, %
    finished = pyqtSignal(dict, dict)    # images{key:QImage}, arrays{key:np.ndarray}
    failed = pyqtSignal(str)             # error text

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.path = None
        self.elev = None
        self.h = self.w = None
        self.lat1d = None
        self.rock_map = None
        self.rock_types = None
        self.rock_colors: Optional[Tuple[Optional[Tuple[int, int, int]], ...]] = None

        # settings
        self.params = dict(
            cellsize=10.0, z_min=0.0, z_max=3000.0,
            sea_level_m=0.0,
            lapse_rate_c_per_km=6.5, t_equator_c=30.0, t_pole_c=0.0,
            coast_decay_km=1.75, orographic_alpha=4.0, shadow_beta=0.15,
            # Advanced rain shadow controls (match CLI defaults)
            shadow_max_distance_km=400.0,
            shadow_decay_km=150.0,
            shadow_height_threshold_m=150.0,
            shadow_strength=1.0,
            precip_lat_pattern="two_bands",
            prevailing_wind_model="three_cell",
            temperature_pattern="polar",
            temperature_gradient_azimuth_deg=0.0,
            precip_gradient_azimuth_deg=0.0,
            constant_wind_azimuth_deg=0.0,
            svf_dirs=16, svf_radius=100.0,
            tpi_radii=[25.0, 100.0],
            biome_mixing=1, use_random_biomes=False,
            flowacc_texture=None,
            albedo_mode="physical",
            deposition_texture=None
        )
        # caches
        self.cache = {}
        # dependency buckets for invalidation
        self._dirty_all()

    # -------- Invalidation helpers --------
    def _dirty_all(self):
        self.cache.clear()

    def _dirty_core(self):
        # everything derived from elevation / cellsize
        keep = {}
        self.cache = keep  # drop all
    def _dirty_ocean_only(self):
        for k in ["ocean", "coastline", "d2coast_m"]:
            self.cache.pop(k, None)
        # Biome & foliage depend on ocean/coast distance
        for k in ["biome_id", "biome_rgb", "albedo_rgb", "albedo_continuous_rgb", "foliage_rgb", "P_mm", "AI", "AET", "PET", "temp_c"]:
            self.cache.pop(k, None)

    def _dirty_climate(self):
        for k in ["wind_u","wind_v","dir_s","P_mm","PET","AET","AI","temp_c","foliage_rgb",
                  "biome_id","biome_rgb","albedo_rgb","albedo_continuous_rgb"]:
            self.cache.pop(k, None)

    def _dirty_svf(self):
        for k in ["svf"]:
            self.cache.pop(k, None)

    def _dirty_tpi(self):
        # remove any tpi_* keys
        for k in list(self.cache.keys()):
            if k.startswith("tpi_"):
                self.cache.pop(k)

    def _dirty_biome_only(self):
        for k in ["biome_id","biome_rgb","albedo_rgb","albedo_continuous_rgb"]:
            self.cache.pop(k, None)

    def _dirty_flowacc(self):
        # Anything depending on flow accumulation or TWI
        for k in ["acc", "twi", "foliage_rgb", "biome_id", "biome_rgb", "albedo_rgb", "albedo_continuous_rgb",
                "forest_density", "groundcover_density", "deposition"]:  # Add "deposition" here
            self.cache.pop(k, None)

    # -------- Public API --------
    def set_settings(self, **kwargs):
        prev = self.params.copy()
        self.params.update(kwargs)

        # decide invalidation scope
        core_changed = any(prev[k] != self.params[k] for k in ["cellsize","z_min","z_max"])
        ocean_changed = (prev["sea_level_m"] != self.params["sea_level_m"])
        climate_changed = any(prev[k] != self.params[k] for k in
                              ["lapse_rate_c_per_km","t_equator_c","t_pole_c",
                               "coast_decay_km","orographic_alpha","shadow_beta",
                               "shadow_max_distance_km","shadow_decay_km",
                               "shadow_height_threshold_m","shadow_strength",
                               "precip_lat_pattern","prevailing_wind_model",
                               "temperature_pattern","temperature_gradient_azimuth_deg",
                               "precip_gradient_azimuth_deg","constant_wind_azimuth_deg"])
        svf_changed = any(prev[k] != self.params[k] for k in ["svf_dirs","svf_radius"])
        tpi_changed = (prev["tpi_radii"] != self.params["tpi_radii"])
        biome_changed = any(prev[k] != self.params[k] for k in ["biome_mixing","use_random_biomes"])
        flowacc_changed = (prev.get("flowacc_texture") != self.params.get("flowacc_texture"))
        albedo_changed = (prev.get("albedo_mode") != self.params.get("albedo_mode") or
                        prev.get("deposition_texture") != self.params.get("deposition_texture"))

        if core_changed:
            self._dirty_core()
        if ocean_changed:
            self._dirty_ocean_only()
        if climate_changed:
            self._dirty_climate()
        if svf_changed:
            self._dirty_svf()
        if tpi_changed:
            self._dirty_tpi()
        if biome_changed:
            self._dirty_biome_only()
        if flowacc_changed:
            self._dirty_flowacc()
        if albedo_changed:
            self.cache.pop("albedo_rgb", None)
            self.cache.pop("deposition", None)

    def load_heightmap_path(self, path):
        p = self.params
        self.path = path
        elev, in_bit_depth = load_heightmap(path, p["z_min"], p["z_max"])
        self.elev = elev.astype(np.float32)
        self.h, self.w = self.elev.shape
        self.lat1d = latitude_degrees(self.h)
        self._dirty_all()

    def inject_deposition_map(self, deposition_map: np.ndarray):
        """Inject a deposition map from external source (e.g., erosion simulation)."""
        if deposition_map is not None:
            self.cache["deposition"] = np.asarray(deposition_map, dtype=np.float32)
            # Clear albedo cache to force recomputation
            self.cache.pop("albedo_rgb", None)

    def inject_rock_map(
        self,
        rock_map: Optional[np.ndarray],
        rock_types: Optional[Iterable[str]] = None,
        rock_colors: Optional[Sequence[Optional[Sequence[int]]]] = None,
    ):
        """Inject a rock type map used for material-aware albedo."""
        if rock_map is None:
            self.rock_map = None
            self.rock_types = None
            self.rock_colors = None
            self.cache.pop("albedo_rgb", None)
            return

        arr = np.asarray(rock_map, dtype=np.int32)
        if self.elev is not None and arr.shape != self.elev.shape:
            raise ValueError("Rock map must match the elevation grid shape.")

        self.rock_map = np.ascontiguousarray(arr)
        if rock_types is not None:
            self.rock_types = tuple(str(name) for name in rock_types)
        else:
            self.rock_types = None

        if rock_colors is not None:
            normalized: list[Optional[Tuple[int, int, int]]] = []
            for entry in rock_colors:
                if entry is None:
                    normalized.append(None)
                    continue
                try:
                    color_tuple = tuple(int(max(0, min(255, float(c)))) for c in entry[:3])  # type: ignore[arg-type]
                    if len(color_tuple) == 3:
                        normalized.append(color_tuple)  # type: ignore[list-item]
                        continue
                except (TypeError, ValueError):
                    pass
                normalized.append(None)
            self.rock_colors = tuple(normalized)
        else:
            self.rock_colors = None

        self.cache.pop("albedo_rgb", None)

    # ---- on-demand compute primitives ----
    def _need(self, key, fn):
        if key in self.cache: return self.cache[key]
        val = fn()
        self.cache[key] = val
        return val

    def get_ocean_masks(self):
        def _o():
            oce = compute_ocean_mask(self.elev, self.params["z_min"], self.params["z_max"], self.params["sea_level_m"])
            return oce
        ocean = self._need("ocean", _o)
        def _c():
            return compute_coastline_mask(ocean)
        coast = self._need("coastline", _c)
        def _d():
            d = distance_to_mask(coast, self.params["cellsize"])
            d[ocean] = 0.0
            return d
        d2 = self._need("d2coast_m", _d)
        return ocean, coast, d2

    def get_slope_aspect_normals(self):
        key = "slope_aspect"
        if key not in self.cache:
            s, a = compute_slope_aspect(self.elev, self.params["cellsize"])
            self.cache[key] = (s, a)
        slope_deg, aspect_deg = self.cache[key]
        normals = self._need("normals", lambda: compute_normals(self.elev, self.params["cellsize"]))
        # gradients for climate
        if "grad" not in self.cache:
            dzdx = np.gradient(self.elev, self.params["cellsize"], axis=1)
            dzdy = np.gradient(self.elev, self.params["cellsize"], axis=0)
            self.cache["grad"] = (dzdx.astype(np.float32), dzdy.astype(np.float32))
        return slope_deg, aspect_deg, normals

    def get_curvature(self):
        return self._need("curv", lambda: compute_laplacian_curvature(self.elev, self.params["cellsize"]))

    def get_tpi(self, r_m):
        tag = f"tpi_{int(r_m)}m"
        if tag in self.cache: return self.cache[tag]
        radius_px = max(1, int(round(r_m / self.params["cellsize"])))
        t = compute_tpi(self.elev, radius_px)
        self.cache[tag] = t
        return t

    def get_flowacc(self):
        def _acc():
            tex = self.params.get("flowacc_texture")
            if tex and os.path.exists(tex):
                # Load external texture as raw values, resize to elevation shape
                arr = load_scalar_texture(tex, target_shape=self.elev.shape)
                return arr.astype(np.float32)
            return d8_flow_accumulation(self.elev, self.params["cellsize"], resolve_pits='carve')
        return self._need("acc", _acc)

    def get_twi(self):
        slope_deg, _, _ = self.get_slope_aspect_normals()
        acc = self.get_flowacc()
        return self._need("twi", lambda: compute_twi(acc, slope_deg, self.params["cellsize"]))

    def get_svf(self):
        p = self.params
        return self._need("svf", lambda: compute_svf(self.elev, p["cellsize"], dirs=p["svf_dirs"], radius_m=p["svf_radius"]))

    def get_climate(self):
        p = self.params
        ocean, coast, d2coast = self.get_ocean_masks()
        slope_deg, aspect_deg, _ = self.get_slope_aspect_normals()
        dzdx, dzdy = self.cache["grad"]

        # winds
        if "wind_u" not in self.cache or "wind_v" not in self.cache:
            if p["prevailing_wind_model"] == "three_cell":
                u_arr, v_arr = prevailing_wind_3cell(self.lat1d)
            else:
                u_arr, v_arr = prevailing_wind(
                    self.lat1d,
                    azimuth_deg=p["constant_wind_azimuth_deg"],
                )
            self.cache["wind_u"] = u_arr
            self.cache["wind_v"] = v_arr
        u = self.cache["wind_u"]
        v = self.cache["wind_v"]

        dir_s = self._need("dir_s", lambda: directional_slope(dzdx, dzdy, u, v))
        temp = self._need("temp_c", lambda: temperature_from_lat_elev(
            self.lat1d,
            self.elev,
            p["lapse_rate_c_per_km"],
            p["t_equator_c"],
            p["t_pole_c"],
            pattern=p["temperature_pattern"],
            gradient_azimuth_deg=p["temperature_gradient_azimuth_deg"],
        ))
        P = self._need("P_mm", lambda: precipitation_orographic_advanced(
            precipitation_lat_bands(
                self.lat1d,
                pattern=p["precip_lat_pattern"],
                width=self.w,
                gradient_azimuth_deg=p["precip_gradient_azimuth_deg"],
            ),
            self.elev, u, v, dzdx, dzdy,
            d2coast, p["cellsize"], alpha=p["orographic_alpha"], beta=p["shadow_beta"],
            coast_decay_m=p["coast_decay_km"]*1000.0, coast_min_frac=0.35,
            # Advanced rain shadow tuning
            shadow_max_distance_km=p["shadow_max_distance_km"],
            shadow_decay_km=p["shadow_decay_km"],
            shadow_height_threshold_m=p["shadow_height_threshold_m"],
            shadow_strength=p["shadow_strength"]
        ))
        PET = self._need("PET", lambda: potential_evapotranspiration(temp, self.lat1d, k=20.0))
        AET = self._need("AET", lambda: actual_evapotranspiration(P, PET))
        AI = self._need("AI", lambda: P / (PET + 1e-6))

        return dict(ocean=ocean, coast=coast, d2coast=d2coast, dir_s=dir_s,
                    temp_c=temp, precip_mm=P, PET=PET, AET=AET, AI=AI,
                    slope_deg=slope_deg, aspect_deg=aspect_deg, u=u, v=v)

    def get_biome(self):
        cl = self.get_climate()
        twi = self.get_twi()
        tpi = self.get_tpi(50.0)
        p = self.params
        def _run():
            bid, brgb = classify_biomes_advanced(
                self.elev, p["sea_level_m"], cl["temp_c"], cl["precip_mm"], cl["PET"], twi,
                cl["slope_deg"], cl["aspect_deg"], tpi, cl["d2coast"], self.lat1d,
                cl["u"], cl["v"], mixing_radius=p["biome_mixing"],
                use_probabilistic=p["use_random_biomes"]
            )
            return bid, brgb
        if "biome_id" not in self.cache or "biome_rgb" not in self.cache:
            bid, brgb = _run()
            self.cache["biome_id"] = bid
            self.cache["biome_rgb"] = brgb
        return self.cache["biome_id"], self.cache["biome_rgb"]

    def get_albedo(self):
        bid, _ = self.get_biome()
        albedo_mode = self.params.get("albedo_mode", "physical")
        
        if albedo_mode == "physical":
            # Get or use cached deposition map
            deposition = self.cache.get("deposition")
            if deposition is None:
                # Check if there's a deposition texture to load
                dep_tex = self.params.get("deposition_texture")
                if dep_tex and os.path.exists(dep_tex):
                    deposition = load_scalar_texture(dep_tex, target_shape=self.elev.shape)
                    deposition = (deposition - 0.5) * 2.0  # Convert from 0-1 to -1 to 1
                else:
                    # Use neutral values if no deposition data
                    deposition = np.zeros_like(self.elev)
                self.cache["deposition"] = deposition
            
            # Get flow mask (wetness)
            twi = self.get_twi()
            flow_mask = np.clip((twi - 4.5) / 6.0, 0.0, 1.0)
            
            # Get vegetation densities
            forest_density, groundcover_density = self.get_foliage_densities()
            
            # Get foliage color
            foliage_rgb = self.get_foliage()
            
            # Get slope
            slope_deg, _, _ = self.get_slope_aspect_normals()
            
            # Get ocean mask
            ocean, _, _ = self.get_ocean_masks()
            
            # Compute physical albedo
            return self._need("albedo_rgb", lambda: compute_terrain_albedo_physical(
                biome_id=bid,
                slope_deg=slope_deg,
                deposition_map=deposition,
                flow_mask=flow_mask,
                forest_density=forest_density,
                groundcover_density=groundcover_density,
                foliage_rgb=foliage_rgb,
                ocean_mask=ocean,
                rock_map=self.rock_map,
                rock_types=self.rock_types,
                rock_colors=self.rock_colors,
                angle_of_repose=35.0,
                cliff_threshold=45.0,
            ))
        else:
            # Original simple biome-based albedo
            return self._need("albedo_rgb", lambda: compute_terrain_albedo_rgb(bid))

    def get_albedo_continuous(self):
        bid, _ = self.get_biome()
        cl = self.get_climate()
        slope_deg, _, _ = self.get_slope_aspect_normals()
        twi = self.get_twi()
        return self._need(
            "albedo_continuous_rgb",
            lambda: compute_terrain_albedo_continuous(
                biome_id=bid,
                slope_deg=slope_deg,
                twi=twi,
                temp_c=cl["temp_c"],
                precip_mm=cl["precip_mm"],
                pet_mm=cl["PET"],
                aridity_index=cl["AI"],
                dist_coast_m=cl["d2coast"],
                latitude_deg=self.lat1d,
                ocean_mask=cl["ocean"],
            ),
        )

    def get_foliage(self):
        cl = self.get_climate()
        svf = self.cache.get("svf", None)  # optional
        tpi_small = self.get_tpi(25.0)
        p = self.params
        return self._need("foliage_rgb", lambda: compute_foliage_color_rgb(
            elev=self.elev, ocean=cl["ocean"], temp_c=cl["temp_c"], precip_mm=cl["precip_mm"],
            pet_mm=cl["PET"], twi=self.cache.get("twi", None),
            slope_deg=cl["slope_deg"], aspect_deg=cl["aspect_deg"],
            dist_coast_m=cl["d2coast"], lat_deg_1d=self.lat1d, svf=svf,
            tpi_small=tpi_small, cellsize=p["cellsize"]
        ))

    def get_foliage_densities(self):
        cl = self.get_climate()
        svf = self.cache.get("svf", None)
        tpi_small = self.get_tpi(25.0)
        p = self.params
        key_f = "forest_density"
        key_g = "groundcover_density"
        if key_f in self.cache and key_g in self.cache:
            return self.cache[key_f], self.cache[key_g]
        fden, gden = compute_foliage_densities(
            elev=self.elev, ocean=cl["ocean"], temp_c=cl["temp_c"], precip_mm=cl["precip_mm"],
            pet_mm=cl["PET"], twi=self.cache.get("twi", None), slope_deg=cl["slope_deg"],
            aspect_deg=cl["aspect_deg"], dist_coast_m=cl["d2coast"], lat_deg_1d=self.lat1d,
            svf=svf, tpi_small=tpi_small, cellsize=p["cellsize"]
        )
        self.cache[key_f] = fden
        self.cache[key_g] = gden
        return fden, gden

    # ---- main entry used by Worker ----
    def compute_selected(self, selections, cancel_callback=None):
        """selections: list of keys to compute"""
        if self.elev is None:
            raise RuntimeError("No heightmap loaded.")
        out_images = {}
        out_arrays = {}

        tot = len(selections)
        for idx, sel in enumerate(selections, 1):
            if callable(cancel_callback) and cancel_callback():
                raise RuntimeError("Execution cancelled.")
            self.progress.emit(f"Computing {sel}...", int(100*idx/tot))

            if sel == "elevation":
                out_arrays["elevation"] = self.elev
                out_images["elevation"] = scalar_to_qimage(self.elev)

            elif sel == "ocean":
                ocean, coast, d2 = self.get_ocean_masks()
                out_arrays["ocean"] = ocean
                out_arrays["coastline"] = coast
                out_arrays["dist2coast_m"] = d2
                out_images["ocean"] = scalar_to_qimage(ocean.astype(np.float32), lo=0, hi=1)
                out_images["coastline"] = scalar_to_qimage(coast.astype(np.float32), lo=0, hi=1)
                out_images["dist2coast_m"] = scalar_to_qimage(d2)

            elif sel == "slope":
                slope_deg, aspect_deg, normals = self.get_slope_aspect_normals()
                out_arrays["slope_deg"] = slope_deg
                out_images["slope_deg"] = scalar_to_qimage(slope_deg, lo=0, hi=60)

            elif sel == "aspect":
                slope_deg, aspect_deg, normals = self.get_slope_aspect_normals()
                out_arrays["aspect_deg"] = aspect_deg
                out_images["aspect_deg"] = scalar_to_qimage(aspect_deg, lo=0, hi=360)

            elif sel == "normal":
                _, _, normals = self.get_slope_aspect_normals()
                out_arrays["normal"] = normals
                out_images["normal"] = normal_to_qimage(normals)

            elif sel == "curvature":
                curv = self.get_curvature()
                out_arrays["curvature"] = curv
                out_images["curvature"] = scalar_to_qimage(curv)

            elif sel.startswith("tpi"):
                # "tpi@25", "tpi@100"
                r_m = float(sel.split("@")[1])
                t = self.get_tpi(r_m)
                key = f"tpi_{int(r_m)}m"
                out_arrays[key] = t
                out_images[key] = scalar_to_qimage(t)

            elif sel == "flowacc":
                acc = self.get_flowacc()
                out_arrays["flowacc"] = acc
                out_images["flowacc_log"] = scalar_to_qimage(np.log1p(acc))

            elif sel == "twi":
                twi = self.get_twi()
                out_arrays["twi"] = twi
                out_images["twi"] = scalar_to_qimage(twi, lo=0, hi=15)

            elif sel == "svf":
                svf = self.get_svf()
                out_arrays["svf"] = svf
                out_images["svf"] = scalar_to_qimage(svf, lo=0, hi=1)

            elif sel == "climate":
                cl = self.get_climate()
                for k in ["temp_c","precip_mm","PET","AET","AI","dir_s"]:
                    out_arrays[k] = cl[k]
                out_images["temp_c"] = scalar_to_qimage(cl["temp_c"], lo=-30, hi=35)
                out_images["precip_mm"] = scalar_to_qimage(cl["precip_mm"])
                out_images["PET"] = scalar_to_qimage(cl["PET"])
                out_images["AET"] = scalar_to_qimage(cl["AET"])
                out_images["AI"] = scalar_to_qimage(cl["AI"], lo=0, hi=2)

            elif sel == "biome":
                bid, brgb = self.get_biome()
                out_arrays["biome_id"] = bid
                out_arrays["biome_rgb"] = brgb
                out_images["biome_id"] = scalar_to_qimage(bid, lo=0, hi=len(BIOME_TABLE)-1)
                out_images["biome_map"] = rgb_to_qimage(brgb)

            elif sel == "albedo":
                alb = self.get_albedo()
                out_arrays["albedo_rgb"] = alb
                out_images["terrain_albedo"] = rgb_to_qimage(alb)

            elif sel == "albedo_continuous":
                alb_cont = self.get_albedo_continuous()
                out_arrays["albedo_continuous_rgb"] = alb_cont
                out_images["terrain_albedo_continuous"] = rgb_to_qimage(alb_cont)

            elif sel == "foliage":
                frgb = self.get_foliage()
                out_arrays["foliage_rgb"] = frgb
                out_images["foliage_color"] = rgb_to_qimage(frgb)

            elif sel == "forest_density":
                fden, gden = self.get_foliage_densities()
                out_arrays["forest_density"] = fden
                out_images["forest_density"] = scalar_to_qimage(fden, lo=0, hi=1)

            elif sel == "groundcover_density":
                fden, gden = self.get_foliage_densities()
                out_arrays["groundcover_density"] = gden
                out_images["groundcover_density"] = scalar_to_qimage(gden, lo=0, hi=1)

        self.finished.emit(out_images, out_arrays)

# -----------------------
# Worker for thread pool
# -----------------------
