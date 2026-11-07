# Test fixture sources

All portraits are works of the U.S. federal government (NASA, U.S. Senate) and are
in the **public domain**. They were downloaded from Wikimedia Commons as 500 px
thumbnails, then resized and transformed by `scripts/make_fixtures.py`.

| Fixture name | Commons file |
|---|---|
| aldrin | File:Buzz_Aldrin.jpg |
| armstrong | File:Neil_Armstrong_pose.jpg |
| glenn | File:John_Glenn_Low_Res.jpg |
| hadfield | File:Chris_Hadfield_2011.jpg |
| jemison | File:Mae_Carol_Jemison.jpg |
| ochoa | File:Ellen_Ochoa.jpg |
| ride | File:Sally_Ride_(1984).jpg |
| whitson | File:Peggy_Whitson.jpg |

The `real`/`fake` folder labels are arbitrary. These are all real photos, and the
labels exist only to exercise the pipeline. The `ffpp/` videos are short pans
over the portraits, laid out like FaceForensics++. They contain no FF++ data.

Edge cases covered: truncated JPEG (`corrupt.jpg`), zero-byte file (`empty.jpg`),
RGBA PNG, 16-bit grayscale PNG, 8-bit grayscale JPEG, CMYK JPEG, EXIF orientation 6,
two faces in one image, a face below 64 px, an image with no face, a near-duplicate
(recompressed and resized) image, and an unreadable video (`999.mp4`).
