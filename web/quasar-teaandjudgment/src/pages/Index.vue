<template>
  <q-page class="flex flex-center">
    <q-img
          :src="url"
          :ratio="1"
          width="30%"
    />
    <q-file filled bottom-slots v-model="imageUpload" accept="image/*" @input="onFileChange" label="Upload image" style="max-width: 400px" counter>
        <template v-slot:prepend>
          <q-icon name="cloud_upload" @click.stop />
        </template>
        <template v-slot:append>
          <q-icon name="close" @click.stop="model = null" class="cursor-pointer" />
        </template>

        <template v-slot:hint>
          Field hint
        </template>
      </q-file>
  </q-page>
</template>

<script>
import axios from 'axios'

export default {
  data () {
    return {
      imageUpload: [],
      url: null
    }
  },
  methods: {
    onFileChange (e) {
      const file = e
      this.url = URL.createObjectURL(file)
      const formData = new FormData()
      formData.append('file.jpg', file)
      console.log('>> formData >> ', formData)

      // You should have a server side REST API
      axios.post('http://127.0.0.1:5000/upload',
        formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      ).then(function () {
        console.log('SUCCESS!!')
      })
        .catch(function () {
          console.log('FAILURE!!')
        })
    }

  }

}

</script>
