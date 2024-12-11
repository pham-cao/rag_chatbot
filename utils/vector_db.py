from qdrant_client import QdrantClient
from decouple import config
from qdrant_client.models import *

from uuid import uuid4

QDRANT_HOST = config('QDRANT_HOST')
QDRANT_POST = config('QDRANT_PORT')

client_db = QdrantClient(host=QDRANT_HOST,
                         port=QDRANT_POST)


def create_new_collection(collection_name, description):
    client_db.create_collection(collection_name=collection_name,
                                vectors_config=VectorParams(size=768,
                                                            distance=Distance.COSINE))
    client_db.upsert(
        collection_name='meta_collection',
        points=[
            models.PointStruct(
                id=str(uuid4()),
                vector=[0] * 1536,
                payload={
                    "name": collection_name,
                    "descriptions": description,
                },
            )])


def delete_collection(collection_name,point_ids):
    client_db.delete_collection(collection_name=collection_name)
    client_db.delete(collection_name='meta_collection',points_selector=PointIdsList(
                points=[point_ids]
            ),)


def get_list_collection_names():
    scroll = client_db.scroll(collection_name='meta_collection', with_payload=True, with_vectors=False)
    data = []
    for item in scroll:
        if item:
            for i in item:
                data.append({'id': i.id,
                             'name': i.payload.get('name'),
                             'description': i.payload.get('descriptions')})
    if len(data) == 0:
        return None
    else:
        return data


