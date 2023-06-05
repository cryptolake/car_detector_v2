#!/usr/bin/env python3
# bot.py
import os
import sys
import requests
import discord
from detector import Pred
from dotenv import load_dotenv
from PIL import ImageDraw, ImageFont

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'{client.user.name} has connected to Discord!')

@client.event
async def on_message(message):
    if client.user.mentioned_in(message):
        if len(message.attachments) > 0:
            for attach in message.attachments:
                try:
                    top = int(message.content.split()[1])
                except Exception:
                    await message.channel.send("Please Provide the number of top Probabilities.")
                    break
                url = attach.url
                image = requests.get(url, stream=True).raw
                prediction = Pred(image, 50, 0.9, top)
                img_rec = ImageDraw.Draw(prediction.image)
                cars = prediction.prediction
                await message.channel.send(f"Found {len(cars)} cars.")
                if len(cars) > 0:
                    for car in cars:
                        box = car.box
                        x0, x1, y0, y1 = box
                        img_rec.rectangle(((x0, y0), (x1, y1)), outline='Red', width=3)
                        cr = car.predictions[0]
                        img_rec.text((x0, y0), text=cr.__str__(), 
                                     font=ImageFont.truetype("arial.ttf", size=int((x1-x0)//25)))
                        await message.channel.send("\n-----------------------------------------------------------------------\n")
                        for i, cr in enumerate(car.predictions):
                            if i != 0:
                                await message.channel.send('OR')
                            await message.channel.send(cr.__str__())
                    await message.channel.send("\n-----------------------------------------------------------------------\n")
                    prediction.image.save('image.png')
                    await message.channel.send(file=discord.File('image.png'))

        else:
            await message.channel.send("send an attachement.")

client.run(TOKEN)
