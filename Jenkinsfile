pipeline {
    agent any

    stages {
        stage('Trigger CodeBuild') {
            steps {
                script {
                    def codebuildProjectName = 'aidocker'
                    def awsRegion = env.AWS_REGION

                    sh """
                    aws codebuild start-build --project-name ${codebuildProjectName} --region ${awsRegion}
                    """
                }
            }
        }
    }
}