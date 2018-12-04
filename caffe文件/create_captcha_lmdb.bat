.\build\tools\Release\convert_imageset.exe --resize_height=227 --resize_width=227 --backend="lmdb" --shuffle .\data\captcha\captcha_train\ .\data\captcha\captcha_train.txt .\data\captcha\captcha_train_lmdb
echo.
.\build\tools\Release\convert_imageset.exe --resize_height=227 --resize_width=227 --backend="lmdb" --shuffle .\data\captcha\captcha_val\ .\data\captcha\captcha_val.txt .\data\captcha\captcha_val_lmdb
echo.
pause

