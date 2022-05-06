<template>
  <div id="app">

    <h1>{{ title }}</h1>
    <VueApexCharts width="600" type="line" :options="options" :series="series"></VueApexCharts>
  </div>
</template>

<script>
import VueApexCharts from 'vue-apexcharts';

import io from 'socket.io-client';

export default {
  name: 'App',
  components: {
    VueApexCharts
  },
  methods: {
    updateChart(data) {

      this.series = [{
        data: data
      }]
    }
  },
  data: function () {
    return {
      data: '',
      title: '',
      options: {
        chart: {
          id: 'vuechart-example'
        },
        xaxis: {
          // categories: [1,2,3]
        }
      },
      series: [{
        name: 'series-1',
        data: []
      }]

    };
  },
  created() {
    // test websocket connection
    const socket = io.connect('http://127.0.0.1:5000');

    // getting data from server
    // eslint-disable-next-line
    socket.on('connect', function () {
      console.error('connected to webSocket');
      //sending to server
      socket.emit('fuck', {data: 'I\'m connected!'});
    });

    // we have to use the arrow function to bind this in the function
    // so that we can access Vue  & its methods
    socket.on('something', (data) => {
      if (this.series[0].data.length) {
        let oldData = this.series[0].data
        oldData.push(data['data'])

        this.updateChart(oldData)
      } else {
        this.series[0].data = data['data'];
        this.title = data['title'];
      }

    });
  },
};
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
