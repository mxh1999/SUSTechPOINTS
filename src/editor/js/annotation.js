
import * as THREE from 'three';
import { jsonrpc } from './jsonrpc.js';
import { globalObjectCategory } from './obj_cfg.js';
import { saveWorldList } from './save.js';
import { intersect } from './util.js';

function Annotation (sceneMeta, world, frameInfo) {
  this.world = world;
  this.data = this.world.data;
  // this.coordinatesOffset = this.world.coordinatesOffset;
  this.boxes_load_time = 0;
  this.frameInfo = frameInfo;

  this.modified = false;
  this.setModified = function () {
    this.modified = true;

    if (window.pointsGlobalConfig.autoSave) {
      saveWorldList([this.world]);
    }
  };
  this.resetModified = function () { this.modified = false; };

  this.sort_boxes = function () {
    this.boxes = this.boxes.sort(function (x, y) {
      return x.position.y - y.position.y;
    });
  };
  this.findBoxByTrackId = function (id) {
    if (this.boxes) {
      const box = this.boxes.find(function (x) {
        return x.obj_track_id == id;
      });
      return box;
    }

    return null;
  };

  this.findIntersectedBoxes = function (box) {
    return this.boxes.filter(b => b != box).filter(b => intersect(box, b));
  };

  this.preload = function (onPreloadFinished) {
    this.onPreloadFinished = onPreloadFinished;
    this.load_annotation((boxes) => this.proc_annotation(boxes));
  };

  this.goCmdReceived = false;
  this.webglScene = null;
  this.onGoFinished = null;
  this.go = function (webglScene, onGoFinished) {
    this.webglScene = webglScene;

    if (this.preloaded) {
      // this.boxes.forEach(b=>this.webglScene.add(b));
      if (this.data.cfg.color_obj != 'no') {
        this.color_boxes();
      }

      if (onGoFinished) { onGoFinished(); }
    } else {
      this.goCmdReceived = true;
      this.onGoFinished = onGoFinished;
    }
  };

  // internal funcs below
  this._afterPreload = function () {
    this.preloaded = true;
    // console.log("annotation preloaded");

    if (this.onPreloadFinished) {
      this.onPreloadFinished();
    }
    if (this.goCmdReceived) {
      this.go(this.webglScene, this.onGoFinished);
    }
  };

  this.unload = function () {
    if (this.boxes) {
      this.boxes.forEach((b) => {
        // this.webglGroup.remove(b);

        if (b.boxEditor) { b.boxEditor.detach(); }
      });
    }
  };

  this.deleteAll = function () {
    this.remove_all_boxes();
    this.destroyed = true;
  };
  this.boxToAnn = function (box) {
    const ann = {
      psr: {
        position: {
          x: box.position.x,
          y: box.position.y,
          z: box.position.z
        },
        scale: {
          x: box.scale.x,
          y: box.scale.y,
          z: box.scale.z
        },
        rotation: {
          x: box.rotation.x,
          y: box.rotation.y,
          z: box.rotation.z
        }
      },
      obj_type: box.obj_type,
      obj_id: String(box.obj_track_id),
      obj_attr: box.obj_attr
      // vertices: vertices,
    };
    return ann;
  };

  this.toBoxAnnotations = function () {
    const anns = this.boxes.filter(b => !b.dontsave).map((b) => {
      // var vertices = pxrToXyz(b.position, b.scale, b.rotation);
      const ann = this.boxToAnn(b);

      if (b.annotator) { ann.annotator = b.annotator; }

      if (b.follows) { ann.follows = b.follows; }
      return ann;
    });

    anns.sort((a, b) => a.obj_id - b.obj_id);

    return anns;
  };

  // to real-world position (no offset)
  this.ann_to_vector_global = function (box) {
    const posG = this.world.lidarPosToUtm(box.position);
    const rotG = this.world.lidarRotToUtm(box.rotation);

    return [
      // posG.x - this.world.coordinatesOffset[0], posG.y-this.world.coordinatesOffset[1], posG.z-this.world.coordinatesOffset[2],
      posG.x, posG.y, posG.z,
      rotG.x, rotG.y, rotG.z,
      box.scale.x, box.scale.y, box.scale.z
    ];
  };

  // real-world position to ann
  this.vector_global_to_ann = function (v) {
    // let posG = new THREE.Vector3(v[0]+this.world.coordinatesOffset[0],
    //                              v[1]+this.world.coordinatesOffset[1],
    //                              v[2]+this.world.coordinatesOffset[2]);
    const posG = new THREE.Vector3(v[0], v[1], v[2]);
    const rotG = new THREE.Euler(v[3], v[4], v[5]);

    const rotL = this.world.utmRotToLidar(rotG);
    const posL = this.world.utmPosToLidar(posG);

    return {
      position: { x: posL.x, y: posL.y, z: posL.z },
      rotation: { x: rotL.x, y: rotL.y, z: rotL.z },
      scale: { x: v[6], y: v[7], z: v[8] }
    };
  };

  // this.vector_to_ann = function(v){
  //     return {
  //         position:{
  //             x:v[0],// + this.coordinatesOffset[0],
  //             y:v[1],// + this.coordinatesOffset[1],
  //             z:v[2],// + this.coordinatesOffset[2],
  //         },

  //         rotation:{
  //             x:v[3],
  //             y:v[4],
  //             z:v[5],
  //         },

  //         scale:{
  //             x:v[6],
  //             y:v[7],
  //             z:v[8],
  //         },

  //     };
  // };

  this.remove_all_boxes = function () {
    if (this.boxes) {
      this.boxes.forEach((b) => {
        this.webglGroup.remove(b);
        this.world.data.dbg.free('box');
        b.geometry.dispose();
        b.material.dispose();
        b.world = null;
        b.boxEditor = null;
      });

      this.boxes = [];
    } else {
      console.error('destroy empty world!');
    }
  };

  this.new_bbox_cube = function (color) {
    const h = 0.5;

    const body = [
      // top
      -h, h, h, h, h, h,
      h, h, h, h, -h, h,
      h, -h, h, -h, -h, h,
      -h, -h, h, -h, h, h,

      // botom
      -h, h, -h, h, h, -h,
      h, h, -h, h, -h, -h,
      h, -h, -h, -h, -h, -h,
      -h, -h, -h, -h, h, -h,

      // vertical lines
      -h, h, h, -h, h, -h,
      h, h, h, h, h, -h,
      h, -h, h, h, -h, -h,
      -h, -h, h, -h, -h, -h,

      // direction
      h, 0, h, 1.5 * h, 0, h
      // h/2, -h, h+0.1,  h, 0, h+0.1,
      // h/2,  h, h+0.1,  h, 0, h+0.1,

      // side direction
      // h, h/2, h,  h, h, 0,
      // h, h/2, -h,  h, h, 0,
      // h, 0, 0,  h, h, 0,

    ];

    this.world.data.dbg.alloc('box');

    const bbox = new THREE.BufferGeometry();
    bbox.setAttribute('position', new THREE.Float32BufferAttribute(body, 3));

    if (!color) {
      color = 0x00ff00;
    }

    /*
        https://threejs.org/docs/index.html#api/en/materials/LineBasicMaterial
        linewidth is 1, regardless of set value.
        */

    const material = new THREE.LineBasicMaterial({ color, linewidth: 1, opacity: this.data.cfg.box_opacity, transparent: true });
    const box = new THREE.LineSegments(bbox, material);

    box.scale.x = 1.8;
    box.scale.y = 4.5;
    box.scale.z = 1.5;
    box.name = 'bbox';
    box.obj_type = 'car';

    // box.computeLineDistances();

    return box;
  };

  this.createCuboid = function (pos, scale, rotation, obj_type, track_id, obj_attr) {
    const mesh = this.new_bbox_cube(parseInt('0x' + globalObjectCategory.getObjCfgByType(obj_type).color.slice(1)));
    mesh.position.x = pos.x;
    mesh.position.y = pos.y;
    mesh.position.z = pos.z;

    mesh.scale.x = scale.x;
    mesh.scale.y = scale.y;
    mesh.scale.z = scale.z;

    mesh.rotation.x = rotation.x;
    mesh.rotation.y = rotation.y;
    mesh.rotation.z = rotation.z;

    mesh.obj_track_id = track_id; // tracking id
    mesh.obj_type = obj_type;
    mesh.obj_attr = obj_attr;
    mesh.obj_local_id = this.get_new_box_local_id();

    mesh.world = this.world;

    return mesh;
  };
  /*
     pos:  offset position, after transformed
    */

  this.add_box = function (pos, scale, rotation, obj_type, track_id, obj_attr) {
    const mesh = this.createCuboid(pos, scale, rotation, obj_type, track_id, obj_attr);

    this.boxes.push(mesh);
    this.sort_boxes();

    this.webglGroup.add(mesh);

    return mesh;
  };

  this.load_box = function (box) {
    this.webglGroup.add(box);
  };

  this.unload_box = function (box) {
    this.webglGroup.remove(box);
  };

  this.remove_box = function (box) {
    this.world.data.dbg.free('box');
    box.geometry.dispose();
    box.material.dispose();
    // selected_box.dispose();
    this.boxes = this.boxes.filter(function (x) { return x != box; });
  };

  this.set_box_opacity = function (box_opacity) {
    this.boxes.forEach(function (x) {
      x.material.opacity = box_opacity;
    });
  };

  this.translate_box_position = function (pos, theta, axis, delta) {
    switch (axis) {
      case 'x':
        pos.x += delta * Math.cos(theta);
        pos.y += delta * Math.sin(theta);
        break;
      case 'y':
        pos.x += delta * Math.cos(Math.PI / 2 + theta);
        pos.y += delta * Math.sin(Math.PI / 2 + theta);
        break;
      case 'z':
        pos.z += delta;
        break;
    }
  };

  this.find_boxes_inside_rect = function (x, y, w, h, camera) {
    const selected_boxes_by_rect = [];

    if (!this.boxes) { return selected_boxes_by_rect; }

    const p = new THREE.Vector3();

    for (let i = 0; i < this.boxes.length; i++) {
      const box_center = this.boxes[i].position;

      const pw = this.world.lidarPosToScene(box_center);
      p.set(pw.x, pw.y, pw.z);
      p.project(camera);
      p.x = p.x / p.z;
      p.y = p.y / p.z;
      // console.log(p);
      if ((p.x > x) && (p.x < x + w) && (p.y > y) && (p.y < y + h)) {
        selected_boxes_by_rect.push(this.boxes[i]);
      }
    }

    console.log('select boxes', selected_boxes_by_rect.length);
    return selected_boxes_by_rect;
  };

  this.proc_annotation = function (boxes) {
    if (this.destroyed) {
      console.error('received boxes after destroyed.');
      return;
    }
    // boxes = this.transformBoxesByEgoPose(boxes);
    // boxes = this.transformBoxesByOffset(boxes);

    // //var boxes = JSON.parse(this.responseText);
    // console.log(ret);
    this.boxes = this.createBoxes(boxes); // create in future world

    this.webglGroup = new THREE.Group();
    this.webglGroup.name = 'annotations';
    this.boxes.forEach(b => this.webglGroup.add(b));

    this.world.webglGroup.add(this.webglGroup);

    this.boxes_load_time = new Date().getTime();
    // console.log(this.boxes_load_time, this.frameInfo.scene, this.frameInfo.frame, "loaded boxes ", this.boxes_load_time - this.create_time, "ms");

    this.sort_boxes();

    this._afterPreload();
  };

  this.load_annotation = function (on_load) {
    if (this.data.cfg.disableLabels) {
      on_load([]);
    } else {
      jsonrpc('/api/load_annotation' + '?scene=' + this.frameInfo.scene + '&frame=' + this.frameInfo.frame).then(ret => {
        if (ret.objs) { on_load(ret.objs); } else { on_load(ret); }
      });
    }
  };

  this.reloadAnnotation = function (done) {
    this.load_annotation(ann => {
      this.reapplyAnnotation(ann, done);
    });
  };

  this.reapplyAnnotation = function (boxes, done) {
    // these boxes haven't attached a world
    // boxes = this.transformBoxesByOffset(boxes);

    // mark all old boxes
    this.boxes.forEach(b => { b.delete = true; });

    const pendingBoxList = [];

    boxes.forEach(nb => { // nb is annotation format, not a true box
      const old_box = this.boxes.find(function (x) {
        return x.obj_track_id == nb.obj_id && x.obj_track_id != '' && nb.obj_id != '' && x.obj_type == nb.obj_type;
      });

      if (old_box) {
        // found
        // update psr
        delete old_box.delete; // unmark delete flag
        old_box.position.set(nb.psr.position.x, nb.psr.position.y, nb.psr.position.z);
        old_box.scale.set(nb.psr.scale.x, nb.psr.scale.y, nb.psr.scale.z);
        old_box.rotation.set(nb.psr.rotation.x, nb.psr.rotation.y, nb.psr.rotation.z);
        old_box.obj_attr = nb.obj_attr;
        old_box.annotator = nb.annotator;
        old_box.changed = false; // clear changed flag.
      } else {
        // not found
        const box = this.createOneBoxByAnn(nb);
        pendingBoxList.push(box);
      }
    });

    // delete removed
    const toBeDelBoxes = this.boxes.filter(b => b.delete);
    toBeDelBoxes.forEach(b => {
      if (b.boxEditor) {
        b.boxEditor.detach('donthide');
      }

      this.webglGroup.remove(b);

      this.remove_box(b);
    });

    pendingBoxList.forEach(b => {
      this.boxes.push(b);
    });

    // todo, restore point color
    // todo, update imagecontext, selected box, ...
    // refer to normal delete operation
    // re-color again
    this.world.lidar.recolorAllPoints();

    this.color_boxes();

    // add new boxes
    pendingBoxList.forEach(b => {
      this.webglGroup.add(b);
    });

    this.resetModified();

    if (done) { done(); }
  };

  this.createOneBoxByAnn = function (annotation) {
    const b = annotation;

    const mesh = this.createCuboid(b.psr.position,
      b.psr.scale,
      b.psr.rotation,
      b.obj_type,
      b.obj_id,
      b.obj_attr);

    if (b.annotator) {
      mesh.annotator = b.annotator;
    }

    if (b.follows) { mesh.follows = b.follows; }

    return mesh;
  };

  this.createBoxes = function (annotations) {
    return annotations.map((b) => {
      return this.createOneBoxByAnn(b);
    });
  };

  this.box_local_id = 0;
  this.get_new_box_local_id = function () {
    const ret = this.box_local_id;
    this.box_local_id += 1;
    return ret;
  };

  this.color_box = function (box) {
    if (this.data.cfg.color_obj == 'category' || this.data.cfg.color_obj == 'no') {
      const color = globalObjectCategory.getColorByType(box.obj_type);
      box.material.color.r = color.x;
      box.material.color.g = color.y;
      box.material.color.b = color.z;
    } else {
      const color = globalObjectCategory.getColorById(box.obj_track_id);
      box.material.color.r = color.x;
      box.material.color.g = color.y;
      box.material.color.b = color.z;
    }
  };

  this.color_boxes = function () {
    this.boxes.forEach(box => {
      this.color_box(box);
    });
  };
}

export { Annotation };
