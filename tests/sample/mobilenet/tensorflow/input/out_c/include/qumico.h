/**
 * Copyright (c) 2019 Pasona Tech Inc. http://pasonatech.co.jp
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
 * to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or  substantial portions of
 * the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * Date Created : 2019-07-30 15:59:36
 * Version      : dev201907
 */

// Qumico Header
#pragma once
//-----------------------------
//compile時のdefine上書きで変更可能
#define QMC_MAX_LAYER_NUM   32  //最大レイヤー数
#define QMC_MAX_LAYER_NAME 256  //レイヤー名の最大文字数

//-----------------------------
//戻り値定義
#define QMC_RET_OK (0)      	//正常終了
#define QMC_RET_ERR (1)     	//エラー

#define QMC_ERR_NO_FILE (2)   	//ファイルがオープンできない
#define QMC_NP_HEADER_ERR (3) 	//numpy読み込み時にヘッダー情報が異常
#define QMC_FREAD_ERR (4)		//fread時にエラー発生

//----------------------------
//ファイルのR/W用定義
#define QMC_MAX_PATH 256
#define QMC_NP_BUF_SIZE 256  	//numpyで使うバッファサイズ

//---------------------------
//型の宣言
typedef long int INT64;
typedef short FIXP16;
typedef signed char FIXP8;
typedef short FP16;
