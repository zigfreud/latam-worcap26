"""Contexto geográfico explícito, independente dos alvos e com CRS verificado."""
import json
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import shapefile
import shapely
import xarray as xr
from pyproj import CRS, Transformer
from shapely.ops import transform

from .experiment import require, sha256
from .spatial import prediction_spatial_features, prepare_spatial_training, spatial_feature_names

GEO_FEATURES = ["ibge_uf_category", "ibge_biome_category", "ibge_vegetation_category"]


def read_polygons(path):
    """Lê zip em memória, sem extrair caminhos; converte SIRGAS2000 para WGS84."""
    with ZipFile(path) as archive:
        bases = [n[:-4] for n in archive.namelist() if n.lower().endswith(".shp")]
        require(len(bases) == 1, "Shapefile ambíguo")
        base = bases[0]
        projection = next(n for n in archive.namelist() if n.startswith(base) and ".prj" in n)
        crs = CRS.from_wkt(archive.read(projection).decode())
        conversion = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        streams = {k: archive.read(base+"."+k) for k in ("shp", "shx", "dbf")}
        records = None
        for encoding in ("utf-8", "cp1252", "latin1"):
            reader = shapefile.Reader(**{k: BytesIO(v) for k, v in streams.items()}, encoding=encoding)
            try:
                records = [r.as_dict() for r in reader.records()]
                break
            except UnicodeDecodeError:
                continue
        require(records is not None, "Codificação DBF")
        polygons = []
        repaired = 0
        for geom in reader.shapes():
            poly = shapely.geometry.shape(geom.__geo_interface__)
            if not poly.is_valid:
                poly = shapely.make_valid(poly)
                repaired += 1
            polygons.append(transform(conversion.transform, poly))
    return records, polygons, {"encoding": encoding, "original_crs": crs.to_string(),
                               "target_crs": "EPSG:4326", "repaired_geometries": repaired}


def assign_categories(polygons, codes, lat, lon):
    """Centros das células; fronteiras usam menor código, sem vizinho mais próximo."""
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    points = shapely.points(xx.ravel(), yy.ravel())
    tree = shapely.STRtree(polygons)
    pairs = tree.query(points, predicate="intersects")
    sentinel = np.iinfo(np.int32).max
    out = np.full(len(points), sentinel, dtype=np.int32)
    if pairs.size:
        np.minimum.at(out, pairs[0], np.asarray(codes, dtype=np.int32)[pairs[1]])
    counts = np.bincount(pairs[0], minlength=len(points)) if pairs.size else np.zeros(len(points))
    out[out == sentinel] = 0
    return out.reshape(yy.shape), int((counts > 1).sum())


def load_context(path, digest, lat, lon):
    require(sha256(path) == digest, "Mapa geográfico mudou")
    with xr.open_dataset(path) as ds:
        require(np.array_equal(ds.lat, lat) and np.array_equal(ds.lon, lon), "Grade geográfica")
        result = ds[GEO_FEATURES].load()
    for key in GEO_FEATURES:
        require(np.issubdtype(result[key].dtype, np.integer) and (result[key] >= 0).all(), "Categoria inválida")
    return result


def prepare_geographic_training(datasets, tp, start, end, points, path, digest):
    base, target, pre = prepare_spatial_training(datasets, tp, start, end, points, "pressure")
    geo = load_context(path, digest, tp.lat.values, tp.lon.values)
    for key in GEO_FEATURES:
        pre[key] = geo[key]
    pre.attrs["geography_sha256"] = digest
    pre.attrs["geography_labels"] = geo.attrs.get("labels_json", "{}")
    pre.attrs["geography_interpretation"] = "Static IBGE reference; NOT observed monthly land cover"
    extra = np.stack([geo[k].values.ravel()[points] for k in GEO_FEATURES], axis=-1)
    rows = np.tile(extra, (len(target)//len(points), 1)).astype(np.float32)
    return np.concatenate([base, rows], axis=1), target, pre


def prediction_geographic_features(datasets, pre, date):
    base = prediction_spatial_features(datasets, pre, date)
    extra = np.stack([pre[k].values.ravel() for k in GEO_FEATURES], axis=-1).astype(np.float32)
    return np.concatenate([base, extra], axis=1)


def geographic_feature_names():
    return spatial_feature_names("pressure")+GEO_FEATURES


def label_json(labels):
    return json.dumps(labels, ensure_ascii=False, sort_keys=True)
