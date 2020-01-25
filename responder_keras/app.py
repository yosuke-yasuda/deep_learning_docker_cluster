import json
import hashlib
import os
import time
import requests
import shutil
import numpy as np
import cv2
import responder
import tensorflow as tf

from data_io import create_img_fromstring
from model import build_model
from main import predict
from logger import Logger
from notification import notify_finish
logger = Logger.generate_logger(__name__)

api = responder.API(
    # this is customized param. see responder_api.py.
    n_bg_queue=int(os.environ.get('N_BG_QUEUE', 1)) 
)

model = build_model()
graph = tf.get_default_graph()

def build_error(resp, e):
    try:
        raise e
    except Exception as e:
        resp.status_code = 500
        resp.media = {'message': 'internal server error', "success": False}
    return resp

class Home:
    def on_get(self, req, resp):
        try:
            resp.headers = {"Content-Type": "application/json; charset=utf-8"}
            resp.content = json.dumps({"success":True})
        except Exception as e:
            logger.error(e, exc_info=True)
            resp = build_error(resp, e)

class Main:
    async def on_post(self, req, resp):
        try:
            resp.headers = {"Content-Type": "application/json; charset=utf-8"}
            params = await req.media(format='files')
            json_params = json.loads(
                params.get("json", "{}")
            )
            
            img_tensor = create_img_fromstring(
                params["img"]["content"]
            )

            def main(*args, **kwargs):
                return predict(*args, **kwargs)

            main_args_dict = dict(
                graph=graph,
                model=model, 
                img_tensor=img_tensor
            )
            @api.background.task
            def main_back(*args, **kwargs):
                try:
                    result = main(*args, **kwargs)
                    notify_finish(
                        endpoint=os.environ.get(
                            "NOTIFICATION_ENDPOINT",
                            None
                        ), 
                        data=result
                    )
                except Exception as e:
                    logger.exception(e, exc_info=True)

            if not json_params.get("async", False):
                result = main(
                    **main_args_dict
                )
                result["success"] = True
                resp.media = result
                resp.content = json.dumps(result)
            else:
                main_back(
                    **main_args_dict
                )
                resp.media = {"success": True}
                resp.content = json.dumps({"success": True})
        except Exception as e:
            logger.error(e, exc_info=True)
            resp = build_error(resp, e)

api.add_route('/', Home)
api.add_route('/main', Main)