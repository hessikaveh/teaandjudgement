<template>
  <q-page class="flex flex-center" style="max-width: 400px">
    <q-ajax-bar ref="bar" :position="position" :reverse="reverse" :size="computedSize" />
    <q-card
      class="my-card text-white"
      style="background: radial-gradient(circle, #35a2ff 0%, #014a88 100%)"
    >
      <q-card-section class="q-pt-none">
        Upload a photo of someone you want to judge and enjoy a roast text extracted from reddit r/RoastMe,
        a unsupervised machine learning algorithm choses the appropriate text for you :)
      </q-card-section>
    </q-card>

    <q-img
          :src="url"
          :ratio="1"
          width="30%"
    />
    <q-file filled bottom-slots v-model="imageUpload" accept="image/*" @input="onFileChange" @click="trigger()" label="Upload image" style="max-width: 400px" counter>
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

      <q-card dark bordered class="bg-grey-9 my-card" style="max-width: 400px">
      <q-card-section>
        <div class="text-h6">Your Roast Text</div>
      </q-card-section>

      <q-separator dark inset />

      <q-card-section>
        {{ category }}
      </q-card-section>
    </q-card>

  </q-page>
</template>

<script>
import axios from 'axios'

export default {
  data () {
    return {
      category: '',
      imageUpload: [],
      url: null,
      position: 'bottom',
      reverse: false,
      size: 4,
      timeouts: []
    }
  },
  computed: {
    computedSize () {
      return this.size + 'px'
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
      axios.post('https://teaandjudgement.herokuapp.com/upload',
        formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      ).then(response => {
        console.log(response)
        const cat = response.data
        this.category = cat
      })
        .catch(function () {
          console.log('FAILURE!!')
        })
    },
    trigger () {
      this.$refs.bar.start()
      setTimeout(() => {
        if (this.$refs.bar) {
          this.$refs.bar.stop()
        }
      }, Math.random() * 5000 + 2000)
    }

  }

}

</script>
