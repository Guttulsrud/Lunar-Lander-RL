import Vue from 'vue'
import App from './App.vue'
import VueApexCharts from 'vue-apexcharts'
import Vuetify from 'vuetify'
import vuetify from './plugins/vuetify'

Vue.use(VueApexCharts)
Vue.use(Vuetify)

Vue.component('apexchart', VueApexCharts)
Vue.config.productionTip = false

new Vue({
  vuetify,
  render: h => h(App)
}).$mount('#app')
