/**
 * @author Filipe Caixeta / http://filipecaixeta.com.br
 * @author Mugen87 / https://github.com/Mugen87
 *
 * Description: A THREE loader for PCD ascii and binary files.
 *
 * Limitations: Compressed binary files are not supported.
 *
 */

import {
  DefaultLoadingManager,
  FileLoader,
  LoaderUtils
} from 'three'

const PCDLoader = function (manager) {
  this.manager = (manager !== undefined) ? manager : DefaultLoadingManager
  this.littleEndian = true
}

function decompressLZF (inData, outLength) {
  const inLength = inData.length
  const outData = new Uint8Array(outLength)
  let inPtr = 0
  let outPtr = 0
  let ctrl
  let len
  let ref
  do {
	  ctrl = inData[inPtr++]
	  if (ctrl < 1 << 5) {
      ctrl++
      if (outPtr + ctrl > outLength) throw new Error('Output buffer is not large enough')
      if (inPtr + ctrl > inLength) throw new Error('Invalid compressed data')
      do {
		  outData[outPtr++] = inData[inPtr++]
      } while (--ctrl)
	  } else {
      len = ctrl >> 5
      ref = outPtr - ((ctrl & 0x1f) << 8) - 1
      if (inPtr >= inLength) throw new Error('Invalid compressed data')
      if (len === 7) {
		  len += inData[inPtr++]
		  if (inPtr >= inLength) throw new Error('Invalid compressed data')
      }

      ref -= inData[inPtr++]
      if (outPtr + len + 2 > outLength) throw new Error('Output buffer is not large enough')
      if (ref < 0) throw new Error('Invalid compressed data')
      if (ref >= outPtr) throw new Error('Invalid compressed data')
      do {
		  outData[outPtr++] = outData[ref++]
      } while (--len + 2)
	  }
  } while (inPtr < inLength)

  return outData
}

PCDLoader.prototype = {

  constructor: PCDLoader,

  load: function (url, onLoad, onProgress, onError, onFileLoaded) {
    const scope = this

    const loader = new FileLoader(scope.manager)
    loader.setPath(scope.path)
    loader.setResponseType('arraybuffer')
    loader.load(url, function (data) {
      try {
        if (onFileLoaded) { onFileLoaded() }
        onLoad(scope.parse(data, url))
      } catch (e) {
        if (onError) {
          onError(e)
        } else {
          throw e
        }
      }
    }, onProgress, onError)
  },

  setPath: function (value) {
    this.path = value
    return this
  },

  parse: function (data, url) {
    const addr = url.split('?')[0].split('.')
    const file_ext = addr[addr.length - 1]

    if (file_ext === 'pcd') { return this.parsePcd(data, url) } else {
      console.log('load', file_ext, 'file')
      return this.parseBin(data, url)
    }
  },

  parseBin: function (data, url) {
    const dataview = new DataView(data, 0)

    const position = []
    const normal = []
    const color = []
    const intensity = []
    // kitti format, xyzi
    const offset = 0

    for (let row = 0; row < data.byteLength / (4 * 4); row += 1) {
      position.push(dataview.getFloat32(row * 16 + 0, this.littleEndian))
      position.push(dataview.getFloat32(row * 16 + 4, this.littleEndian))
      position.push(dataview.getFloat32(row * 16 + 8, this.littleEndian))
      intensity.push(dataview.getFloat32(row * 16 + 12, this.littleEndian))
    }

    return {
      position,
      color,
      normal,
      intensity
    }
  },

  parsePcd: function (data, url) {
    function parseHeader (data) {
      const PCDheader = {}
      const result1 = data.search(/[\r\n]DATA\s(\S*)\s/i)
      const result2 = /[\r\n]DATA\s(\S*)\s/i.exec(data.substr(result1 - 1))

      PCDheader.data = result2[1]
      PCDheader.headerLen = result2[0].length + result1
      PCDheader.str = data.substr(0, PCDheader.headerLen)

      // remove comments

      PCDheader.str = PCDheader.str.replace(/\#.*/gi, '')

      // parse

      PCDheader.version = /VERSION (.*)/i.exec(PCDheader.str)
      PCDheader.fields = /FIELDS (.*)/i.exec(PCDheader.str)
      PCDheader.size = /SIZE (.*)/i.exec(PCDheader.str)
      PCDheader.type = /TYPE (.*)/i.exec(PCDheader.str)
      PCDheader.count = /COUNT (.*)/i.exec(PCDheader.str)
      PCDheader.width = /WIDTH (.*)/i.exec(PCDheader.str)
      PCDheader.height = /HEIGHT (.*)/i.exec(PCDheader.str)
      PCDheader.viewpoint = /VIEWPOINT (.*)/i.exec(PCDheader.str)
      PCDheader.points = /POINTS (.*)/i.exec(PCDheader.str)

      // evaluate

      if (PCDheader.version !== null) { PCDheader.version = parseFloat(PCDheader.version[1]) }

      if (PCDheader.fields !== null) { PCDheader.fields = PCDheader.fields[1].split(' ') }

      if (PCDheader.type !== null) { PCDheader.type = PCDheader.type[1].split(' ') }

      if (PCDheader.width !== null) { PCDheader.width = parseInt(PCDheader.width[1]) }

      if (PCDheader.height !== null) { PCDheader.height = parseInt(PCDheader.height[1]) }

      if (PCDheader.viewpoint !== null) { PCDheader.viewpoint = PCDheader.viewpoint[1] }

      if (PCDheader.points !== null) { PCDheader.points = parseInt(PCDheader.points[1], 10) }

      if (PCDheader.points === null) { PCDheader.points = PCDheader.width * PCDheader.height }

      if (PCDheader.size !== null) {
        PCDheader.size = PCDheader.size[1].split(' ').map(function (x) {
          return parseInt(x, 10)
        })
      }

      if (PCDheader.count !== null) {
        PCDheader.count = PCDheader.count[1].split(' ').map(function (x) {
          return parseInt(x, 10)
        })
      } else {
        PCDheader.count = []

        for (var i = 0, l = PCDheader.fields.length; i < l; i++) {
          PCDheader.count.push(1)
        }
      }

      PCDheader.offset = {}

      let sizeSum = 0

      for (var i = 0, l = PCDheader.fields.length; i < l; i++) {
        if (PCDheader.data === 'ascii' || PCDheader.data === 'ascill') {
          PCDheader.offset[PCDheader.fields[i]] = i
        } else {
          PCDheader.offset[PCDheader.fields[i]] = sizeSum
          sizeSum += PCDheader.size[i] * PCDheader.count[i]
        }
      }

      // for binary only

      PCDheader.rowSize = sizeSum

      return PCDheader
    }

    const textData = LoaderUtils.decodeText(new Uint8Array(data))

    // parse header (always ascii format)

    const PCDheader = parseHeader(textData)

    // parse data

    const position = []
    const normal = []
    const color = []
    const velocity = []
    const intensity = []

    // ascii

    function filterPoint (x, y, z) {
      if (isNaN(x)) { return true }
      if (x == 0 && y == 0 && z == 0) { return true }
      // if (z >=2)
      // 	return true;
    }

    if (PCDheader.data === 'ascii' || PCDheader.data === 'ascill') {
      var offset = PCDheader.offset
      const pcdData = textData.substr(PCDheader.headerLen)
      const lines = pcdData.split('\n')

      var intensity_index = PCDheader.fields.findIndex(n => n === 'intensity')
      var intensity_type = 'F'
      var intensity_size = 4

      if (intensity_index >= 0) {
        intensity_type = PCDheader.type[intensity_index]
        intensity_size = PCDheader.size[intensity_index]
      }

      for (var i = 0, l = lines.length; i < l; i++) {
        if (lines[i] === '') continue

        const line = lines[i].split(' ')

        if (offset.x !== undefined) {
          var x, y, z
          x = parseFloat(line[offset.x])
          y = parseFloat(line[offset.y])
          z = parseFloat(line[offset.z])

          if (filterPoint(x, y, z)) {
            continue
          }

          position.push(x)
          position.push(y)
          position.push(z)
        }

        if (offset.rgb !== undefined) {
          const rgb = parseFloat(line[offset.rgb])
          const r = (rgb >> 16) & 0x0000ff
          const g = (rgb >> 8) & 0x0000ff
          const b = (rgb >> 0) & 0x0000ff
          color.push(r / 255, g / 255, b / 255)
        }

        if (offset.normal_x !== undefined) {
          normal.push(parseFloat(line[offset.normal_x]))
          normal.push(parseFloat(line[offset.normal_y]))
          normal.push(parseFloat(line[offset.normal_z]))
        }

        if (offset.intensity !== undefined) {
          intensity.push(parseInt(line[offset.intensity]))
        }
      }
    }

    // binary

    if (PCDheader.data === 'binary_compressed') {
      const sizes = new Uint32Array(data.slice(PCDheader.headerLen, PCDheader.headerLen + 8))
      const compressedSize = sizes[0]
      const decompressedSize = sizes[1]
      const decompressed = decompressLZF(new Uint8Array(data, PCDheader.headerLen + 8, compressedSize), decompressedSize)
      var dataview = new DataView(decompressed.buffer)

      var offset = PCDheader.offset
      var intensity_index = PCDheader.fields.findIndex(n => n === 'intensity')
      var intensity_type = 'F'
      var intensity_size = 4

      if (intensity_index >= 0) {
        intensity_type = PCDheader.type[intensity_index]
        intensity_size = PCDheader.size[intensity_index]
      }

      const size = {}

      PCDheader.fields.forEach((n, i) => size[n] = PCDheader.size[i])

      for (var i = 0; i < PCDheader.points; i++) {
        if (offset.x !== undefined) {
          if (size.x == 8) {
            position.push(dataview.getFloat64((PCDheader.points * offset.x) + size.x * i, this.littleEndian))
            position.push(dataview.getFloat64((PCDheader.points * offset.y) + size.y * i, this.littleEndian))
            position.push(dataview.getFloat64((PCDheader.points * offset.z) + size.z * i, this.littleEndian))
          } else {
            position.push(dataview.getFloat32((PCDheader.points * offset.x) + size.x * i, this.littleEndian))
            position.push(dataview.getFloat32((PCDheader.points * offset.y) + size.y * i, this.littleEndian))
            position.push(dataview.getFloat32((PCDheader.points * offset.z) + size.z * i, this.littleEndian))
          }
        }

        // if ( offset.rgb !== undefined ) {

        // 	color.push( dataview.getUint8( ( PCDheader.points * ( offset.rgb + 2 ) ) + PCDheader.size[ 3 ] * i ) / 255.0 );
        // 	color.push( dataview.getUint8( ( PCDheader.points * ( offset.rgb + 1 ) ) + PCDheader.size[ 3 ] * i ) / 255.0 );
        // 	color.push( dataview.getUint8( ( PCDheader.points * ( offset.rgb + 0 ) ) + PCDheader.size[ 3 ] * i ) / 255.0 );

        // }

        // if ( offset.normal_x !== undefined ) {

        // 	normal.push( dataview.getFloat32( ( PCDheader.points * offset.normal_x ) + PCDheader.size[ 4 ] * i, this.littleEndian ) );
        // 	normal.push( dataview.getFloat32( ( PCDheader.points * offset.normal_y ) + PCDheader.size[ 5 ] * i, this.littleEndian ) );
        // 	normal.push( dataview.getFloat32( ( PCDheader.points * offset.normal_z ) + PCDheader.size[ 6 ] * i, this.littleEndian ) );

        // }

        if (offset.intensity !== undefined) {
          if (intensity_type == 'U' && intensity_size == 1) {
            intensity.push(dataview.getUint8(PCDheader.points * offset.intensity + size.intensity * i))
          } else if (intensity_type == 'F' && intensity_size == 4) {
            intensity.push(dataview.getFloat32(PCDheader.points * offset.intensity + size.intensity * i, this.littleEndian))
          }
        }
      }
    } else if (PCDheader.data === 'binary') {
      var dataview = new DataView(data, PCDheader.headerLen)
      var offset = PCDheader.offset

      var intensity_index = PCDheader.fields.findIndex(n => n === 'intensity')
      var intensity_type = 'F'
      var intensity_size = 4

      if (intensity_index >= 0) {
        intensity_type = PCDheader.type[intensity_index]
        intensity_size = PCDheader.size[intensity_index]
      }

      const x_index = PCDheader.fields.findIndex(n => n === 'x')
      let x_size = 4
      let x_type = 'F'
      if (x_index >= 0) {
        x_type = PCDheader.type[x_index]
        x_size = PCDheader.size[x_index]
      }

      for (var i = 0, row = 0; i < PCDheader.points; i++, row += PCDheader.rowSize) {
        if (offset.x !== undefined) {
          const getFloat = (x_size == 8) ? dataview.getFloat64.bind(dataview) : dataview.getFloat32.bind(dataview)

          const x = getFloat(row + offset.x, this.littleEndian)
          const y = getFloat(row + offset.y, this.littleEndian)
          const z = getFloat(row + offset.z, this.littleEndian)

          if (filterPoint(x, y, z)) {
            continue
          }

          position.push(x)
          position.push(y)
          position.push(z)
        }

        if (offset.rgb !== undefined) {
          color.push(dataview.getUint8(row + offset.rgb + 2) / 255.0)
          color.push(dataview.getUint8(row + offset.rgb + 1) / 255.0)
          color.push(dataview.getUint8(row + offset.rgb + 0) / 255.0)
        }

        if (offset.normal_x !== undefined) {
          normal.push(dataview.getFloat32(row + offset.normal_x, this.littleEndian))
          normal.push(dataview.getFloat32(row + offset.normal_y, this.littleEndian))
          normal.push(dataview.getFloat32(row + offset.normal_z, this.littleEndian))
        }

        if (offset.vx !== undefined) {
          velocity.push(dataview.getFloat32(row + offset.vx, this.littleEndian))
          velocity.push(dataview.getFloat32(row + offset.vy, this.littleEndian))
          velocity.push(0)
        }

        if (offset.intensity !== undefined) {
          if (intensity_type == 'U' && intensity_size == 1) {
            intensity.push(dataview.getUint8(row + offset.intensity))
          } else if (intensity_type == 'F' && intensity_size == 4) {
            intensity.push(dataview.getFloat32(row + offset.intensity, this.littleEndian))
          }
        }
      }
    }

    return {
      position,
      color,
      normal,
      velocity,
      intensity
    }
  }

}

export { PCDLoader }
