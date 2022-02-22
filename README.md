# notesXtract

Extractor for color highlighted notes from scanned (or photographed) documents with OpenCV and Tesseract.


# get started

To setup do
```
pip install pytesseract
```
```
pip install cv2
```
also you need to have numpy installed the same way. If you clone the repository, keep the temp and the Output directory (or change the paths).

If you want to use another language than English, you should download the respective tesseract package and include the path to your tesseract installation in the file (Line 11). Also you have to change the lang config of tesseract in line 81 and 89.

# short manual

To process your highlighted documents, scan these or take a photo (in Example Folder you see, which quality works best). The better your document picture, the better the output. 

Choose one or more files in filedialogue. 

The documents will be processed and your notes will be written to a txt-file. If you don't want an errorcorrection on your notes, comment out the errormanage function at the end.

After that, you can delete the content of the temp directory (or leave it).


# please notice

How much the notes can be corrected afterwards depends on how well tesserect can read your input file. 

Also the accuracy of your highlighting is essential for processing: Half marked lines/characters are difficult to process. Often the errorcorrection can help, but not always. 

The input pictures should be in good quality for reading. 

Errorcorrecting sometimes (but really rarely) will add words, that aren't really correct. That's because of the use of trigrams, to reconstruct the text and the process of choosing 80% appropriate text sequences. Sometimes this will choose as "best result" an trigram, that continues with a different word than your note. 

If input is absolutely not readable (because of bad picture or bad highlighting etc.) it will raise an "list out of range"-error.

Please keep in mind: It's for notes, not for perfect documents (please be kind) ;-) Thank you!


# to do

Better method for errorcorrection (if not trigram -> don't give back 3 words)
No word duplicates anymore, that follow each other
More comfortable way to turn errorcorrection on/off (GUI?)
Error Warning if input is not readable
Support of PDFs
Maybe an exe-file


# footnotes

The example is from the paper: *Zhuang, Fuzhen, et al. "A comprehensive survey on transfer learning." Proceedings of the IEEE 109.1 (2020): 43-76.*