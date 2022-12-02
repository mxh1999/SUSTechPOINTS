
import { jsonrpc } from './jsonrpc.js';

class ImageRectAnnotation {
  constructor (sceneMeta, frameInfo) {
    this.sceneMeta = sceneMeta;
    this.scene = frameInfo.scene;
    this.frame = frameInfo.frame;

    this.anns = {
      camera: {},
      aux_camera: {}
    };

    this.goCmdReceived = false;
    this.onGoFinished = null;
  }

  preload (onPreloadFinished) {
    this.onPreloadFinished = onPreloadFinished;
    this.loadAll();
  }

  //
  load (cameraType, cameraName) {
    return this.anns[cameraType][cameraName];
  }

  save (cameraType, cameraName, data) {
    this.anns[cameraType][cameraName] = data;
    jsonrpc('/api/save_image_annotation', 'POST', data).then(ret => {
      console.log('saved', ret);
    }).catch(e => {
      window.editor.infoBox.show('Error', 'save failed');
    });
  }

  /// /////////////////

  fetchAnn (cameraType, cameraName) {
    return jsonrpc(`/api/load_image_annotation?scene=${this.scene}&frame=${this.frame}&camera_type=${cameraType}&camera_name=${cameraName}`);
  }

  fetchAll () {
    const aux_cameras = this.sceneMeta.aux_camera.reduce((a,b)=>a+','+b);
    const cameras = this.sceneMeta.camera.reduce((a,b)=>a+','+b);
    return jsonrpc(`/api/load_all_image_annotation?scene=${this.scene}&frame=${this.frame}&cameras=${cameras}&aux_cameras=${aux_cameras}`);
  }

  loadAll () {
    this.fetchAll().then(ret=>{
      this.anns = ret;
      this.preloaded = true;
      if (this.onPreloadFinished) {
        this.onPreloadFinished();
      }
      if (this.goCmdReceived) {
        this.go(this.webglScene, this.onGoFinished);
      }
    })
  }

  loadOneByOne() {
    let annsAsync = [];
    if (this.sceneMeta.aux_camera) {
      annsAsync = annsAsync.concat(this.sceneMeta.aux_camera.map(c => {
        return this.fetchAnn('aux_camera', c).then(ret => {
          this.anns.aux_camera[c] = ret;
        });
      }));
    }

    if (this.sceneMeta.camera) {
      annsAsync = annsAsync.concat(this.sceneMeta.camera.map(c => {
        return this.fetchAnn('camera', c).then(ret => {
          this.anns.camera[c] = ret;
        });
      }));
    }

    Promise.all(annsAsync).then(ret => {
      this.preloaded = true;
      if (this.onPreloadFinished) {
        this.onPreloadFinished();
      }
      if (this.goCmdReceived) {
        this.go(this.webglScene, this.onGoFinished);
      }
    });
  }

  go (webglScene, onGoFinished) {
    if (this.preloaded) {
      if (onGoFinished) { onGoFinished(); }
    } else {
      this.goCmdReceived = true;
      this.onGoFinished = onGoFinished;
    }
  }

  unload () {

  }
}

export { ImageRectAnnotation };
