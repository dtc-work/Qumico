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
