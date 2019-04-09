# elroy
Self descriptive

## On restart

```
$ sudo modprobe v4l2loopback devices=1 exclusive_caps=1 video_nr=2 cap_label="ElroyCam"
$ ./fix_webcams.sh
```

## To run
```
$ python3 test.py
```

## To watch
```
$ DISPLAY=:0 vlc v4l2:///dev/video2
```
