
int load_from_numpy(void *dp, const char *numpy_fname, int size, NUMPY_HEADER *hp)
{

  FILE *fp;
  int ret;
  int size_from_shape;

  assert(dp!=NULL);
  assert(numpy_fname!=NULL);
  assert(size > 0);
  assert(hp!=NULL);

  fp = fopen(numpy_fname, "rb");
  if(fp==NULL) {
    printf("ERROR:cant'open %s\n", numpy_fname);
    return 2;

  }

  ret = np_check_header(fp, hp);
  if(ret != 0) {
    printf("ERROR:numpy header error %s\n", numpy_fname);
    return ret;
  }

  printf("load from %s\n", numpy_fname);
  //np_print_heaer_info(hp);

  //引数のサイズと、numpyヘッダーのサイズを比較
  if(hp->shape[1] == 0) {
      size_from_shape = hp->shape[0];
  } else if (hp->shape[2] == 0) {
      size_from_shape = hp->shape[0] * hp->shape[1];
  } else if  (hp->shape[3] == 0) {
      size_from_shape = hp->shape[0] * hp->shape[1] * hp->shape[2];
  } else {
      size_from_shape = hp->shape[0] * hp->shape[1] *  hp->shape[2] *  hp->shape[3];
  }

  printf("size = %d, size_from_shape = %d\n", size, size_from_shape);

  if(size != size_from_shape) {
    printf("ERROR:numpy header error %s\n", numpy_fname);
    return CQT_NP_HEADER_ERR;
  }

  switch (hp->descr) {
  case CQT_FLOAT32:
      assert(sizeof(float)==4);
      ret = fread(dp, 4, size, fp);
      if (ret != size) {
        return CQT_FREAD_ERR;
      }
      break;
  case CQT_UINT8: //fall through
  case CQT_FIX8:
      ret =  fread(dp, 1, size, fp);
      if (ret != size) {
        return CQT_FREAD_ERR;
      }
      break;

  case CQT_FIX16:   //fall through
  case CQT_FLOAT16:
      ret = fread(dp, 2, size, fp);
      if (ret != size) {
        return CQT_FREAD_ERR;
      }
      break;

  default:
      printf("ERROR:numpy header error dscr = %d\n", hp->descr);
      return CQT_NP_HEADER_ERR;
  }

  fclose(fp);

  return CQT_RET_OK;
}
