for i in *.bmp; do convert ${i} ${i%bmp}jpg; done
