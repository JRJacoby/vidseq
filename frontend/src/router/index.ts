import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'
import ProjectView from '../views/ProjectView.vue'
import VideoPipeline from '../components/VideoPipeline.vue'
import JobsList from '../components/JobsList.vue'
import VideoDetail from '../components/VideoDetail.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView
    },
    {
      path: '/project/:id',
      component: ProjectView,
      children: [
        {
          path: '',
          name: 'pipeline',
          component: VideoPipeline
        },
        {
          path: 'jobs',
          name: 'jobs',
          component: JobsList
        },
        {
          path: 'video/:videoId',
          name: 'video',
          component: VideoDetail
        }
      ]
    }
  ],
})

export default router
