
import { jsonrpc } from './jsonrpc.js'

class ImageRectAnnotation {
  constructor (sceneMeta, frameInfo) {
    this.sceneMeta = sceneMeta
    this.scene = frameInfo.scene
    this.frame = frameInfo.frame
  }

  preload (onPreloadFinished) {
    this.onPreloadFinished = onPreloadFinished
    this.loadAll()
  };

  //
  load (cameraType, cameraName) {
    return this.anns[cameraType][cameraName]
  }

  save (cameraType, cameraName, data) {
    this.anns[cameraType][cameraName] = data
    jsonrpc('/api/save_image_annotation', 'POST', data).then(ret => {
      console.log('saved', ret)
    }).catch(e => {
      window.editor.infoBox.show('Error', 'save failed')
    })
  }

  /// /////////////////

  anns = {
    camera: {},
    aux_camera: {}
  }

  fetchAnn (cameraType, cameraName) {
    return jsonrpc(`/api/load_image_annotation?scene=${this.scene}&frame=${this.frame}&camera_type=${cameraType}&camera_name=${cameraName}`)
  }

  loadAll () {
    let annsAsync = []
    if (this.sceneMeta.aux_camera) {
      annsAsync = annsAsync.concat(this.sceneMeta.aux_camera.map(c => {
        return this.fetchAnn('aux_camera', c).then(ret => {
          this.anns.aux_camera[c] = ret
        })
      }))
    }

    if (this.sceneMeta.camera) {
      annsAsync = annsAsync.concat(this.sceneMeta.camera.map(c => {
        return this.fetchAnn('camera', c).then(ret => {
          this.anns.camera[c] = ret
        })
      }))
    }

    Promise.all(annsAsync).then(ret => {
      this.preloaded = true
      if (this.onPreloadFinished) {
        this.onPreloadFinished()
      }
      if (this.goCmdReceived) {
        this.go(this.webglScene, this.onGoFinished)
      }
    })
  };

  goCmdReceived = false
  onGoFinished = null

  go (webglScene, onGoFinished) {
    if (this.preloaded) {
      if (onGoFinished) { onGoFinished() }
    } else {
      this.goCmdReceived = true
      this.onGoFinished = onGoFinished
    }
  };

  unload () {

  };
}

export { ImageRectAnnotation }
