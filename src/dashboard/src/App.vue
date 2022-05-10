<template>
  <v-app>
    <v-container fluid class="grey lighten-3" style="min-height: 100rem">
      <v-container>
        <v-card class="white pa-12">
          <h1 class="text-center">Training model: {{ title }}</h1>

          <v-row class="mt-12">
            <v-col cols="6" class="pa-12">
              <h2>Score per episode</h2>
              <VueApexCharts type="line" :options="options" :series="series"></VueApexCharts>

            </v-col>
            <v-col cols="6" class="pa-12">
              <h2>All evaluations</h2>
              <VueApexCharts type="line" :options="options2" :series="series2"></VueApexCharts>
            </v-col>
          </v-row>
        </v-card>

      </v-container>
    </v-container>

  </v-app>
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
    },
    updateChart2(data) {

      this.series2 = [{
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
        // yaxis: {
        //   min: -250,
        //   max: 200
        // }

      },
         options2: {
        chart: {
          id: 'vuechart-example'
        },

      },
      series: [{
        name: 'series-1',
        data: []
      }],
      series2: [{
        name: 'series-2',
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

    socket.on('something2', (data) => {
      if (this.series2[0].data.length) {
        let oldData = this.series2[0].data
        oldData.push(data['data'])

        this.updateChart2(oldData)
      } else {
        this.series2[0].data = data['data'];
      }

    });
  },
};
</script>
