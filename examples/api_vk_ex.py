#!/usr/bin/python
# -*- coding: utf-8 -*-

import requests
import urllib

GET_GROUP_ID_MTHD = "https://api.vk.com/method/groups.getById"
GET_PHOTOS_MTHD = "https://api.vk.com/method/photos.get"

s = requests.Session()
groups = ["mipt_no_doubt"]
a = s.get(GET_GROUP_ID_MTHD + "?group_id=" + groups[0])
gid = a.json()["response"][0]["gid"]
r = s.get(GET_PHOTOS_MTHD + "?owner_id=-" + str(gid)+ "&album_id=wall")
print r.json()["response"][0]["src"]

urllib.urlretrieve(r.json()["response"][0]["src"], "vk")