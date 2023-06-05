from base64 import b64encode
from fastapi import FastAPI, Form, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Annotated
from detector import Pred, draw_bbox
from io import BytesIO

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse('home/index.html', {"request": request})

@app.get("/about")
async def about(request: Request):
    return templates.TemplateResponse('home/about.html', {"request": request})

@app.post("/detect")
async def detect(
    request: Request,
    image: Annotated[bytes, File()],
    threshold: Annotated[float, Form()],
    top: Annotated[int, Form()],
    min_size: Annotated[int, Form()]
    ):
    cars = Pred(BytesIO(image), min_size, threshold, top)

    buf = BytesIO()
    img = draw_bbox(cars.image, cars.prediction)
    img.save(buf, format='JPEG')
    byte_im = buf.getvalue()
    return templates.TemplateResponse('result.html',
                                      {"request": request, 'cars': cars.prediction,
                                       'image': b64encode(byte_im).decode("utf-8")})
